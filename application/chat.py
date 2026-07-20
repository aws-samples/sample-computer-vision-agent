# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Multi-Agent Framework Core Orchestration

This module provides the core orchestration logic for the multi-agent framework.
It connects agents with MCP tools and provides both single and multi-agent patterns.

🏗️ FRAMEWORK ARCHITECTURE:
- Single Agent: cv_agent_impl() - Direct CV tool usage for simple deployments
- Multi Agent: create_interaction_agent() - Orchestrates multiple specialist agents
- MCP Server: Provides CV tools via Model Context Protocol
- Agent Prompts: Defined in prompts/ directory for easy configuration

🔄 EXTENSION PATTERNS:

1. Adding New Agents:
   - Create prompts/new_agent.py with SYSTEM_PROMPT and QUERY_TEMPLATE
   - Import: from prompts.new_agent import SYSTEM_PROMPT as NEW_SYSTEM_PROMPT
   - Create agent function following cv_agent_impl() pattern

2. Adding New Tools:
   - Extend aws_cv_mcp_server/cv_tools.py with new tool functions
   - Register tools in aws_cv_mcp_server/server.py
   - Update agent prompts to reference new tools

3. Multi-Agent Coordination:
   - Use create_interaction_agent() as orchestrator
   - Create specialist agents and expose as tools to orchestrator
   - Follow delegation pattern shown in interaction_agent.py

📋 KEY FUNCTIONS:
- cv_agent_impl(): Single-agent CV specialist
- create_interaction_agent(): Multi-agent orchestrator
- get_mcp_tools(): Connects to MCP server for tool access
- get_model(): Initializes the LLM model

🔧 CONFIGURATION:
- Model settings: Configure model_id, model_name variables
- Agent prompts: Edit files in prompts/ directory
- Tool behavior: Modify aws_cv_mcp_server/cv_tools.py
"""

import asyncio
import logging
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import List
from botocore.config import Config
from mcp import StdioServerParameters, stdio_client
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

# Agent Configuration - Simple Direct Imports (HUMAN EDIT HERE)
from prompts.cv_agent import SYSTEM_PROMPT as CV_SYSTEM_PROMPT
import info

#########################################################
# Thread-Safe Global Image Storage
#########################################################

# Global storage for images that need to be displayed
# This allows background agent threads to communicate image data to the main UI thread
_global_image_storage = []
_storage_lock = threading.Lock()

def clear_global_image_storage():
    """Clear the global image storage (call at start of each request)"""
    global _global_image_storage  # pylint: disable=global-variable-not-assigned
    with _storage_lock:
        _global_image_storage.clear()
        logger.info("Cleared global image storage")

def add_image_to_global_storage(image_data):
    """Thread-safe way to add image data from background threads"""
    global _global_image_storage  # pylint: disable=global-variable-not-assigned
    with _storage_lock:
        _global_image_storage.append(image_data)
        logger.info(f"Added image to global storage: {image_data.get('s3_key', 'unknown')}")

def get_images_from_global_storage():
    """Thread-safe way to retrieve and clear image data from main thread"""
    global _global_image_storage  # pylint: disable=global-variable-not-assigned
    with _storage_lock:
        images = _global_image_storage.copy()
        _global_image_storage.clear()
        logger.info(f"Retrieved {len(images)} images from global storage")
        return images

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("chat")
logging.getLogger("botocore").setLevel(logging.WARNING)

model_name = "Claude 4 Sonnet"
model_type = "claude"
debug_mode = "Enable"
model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

models = info.get_model_info(model_name)
reasoning_mode = "Disable"
# SEC: H8 — Only bypass tool consent in non-production environments
if os.environ.get("ENVIRONMENT", "production") != "production":
    os.environ["BYPASS_TOOL_CONSENT"] = "true"


def update(modelName, reasoningMode):
    """Update the model configuration.
    
    Args:
        modelName (str): The name of the model to update to
        reasoningMode (str): The reasoning mode to use
    """
    global model_name, model_id, model_type, reasoning_mode  # pylint: disable=global-statement

    if model_name != modelName:
        model_name = modelName


        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]
        logger.info(f"Updating model to model_name: {model_name}, {model_id}")

    if reasoningMode != reasoning_mode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")


def initiate():
    """Initialize the user session with a unique ID."""
    global userId  # pylint: disable=global-variable-undefined,global-statement
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")


#########################################################
# Strands Agent Model Configuration
#########################################################
def get_model():
    """Get the current model configuration.
    
    Returns:
        dict: The model configuration profile
    """
    profile = models[0]
    logger.info("[DEBUG] === CHAT MODEL DEBUG ===")
    logger.info(f"[DEBUG] Selected model profile: {profile}")
    logger.info(f"[DEBUG] Current AWS_REGION env var: {os.environ.get('AWS_REGION')}")
    logger.info(f"[DEBUG] All available models for '{model_name}': {models}")

    if profile["model_type"] == "nova":
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile["model_type"] == "claude":
        STOP_SEQUENCE = "\n\nHuman:"

    if model_type == "claude":
        maxOutputTokens = 64000  # 4k
    else:
        maxOutputTokens = 5120  # 5k

    maxReasoningOutputTokens = 64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens - 1000)

    if reasoning_mode == "Enable":
        # Configure thinking parameters
        thinking_config = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

        additional_fields = {"thinking": thinking_config}

        # Add interleaved thinking for Claude 4 Sonnet using anthropic_beta parameter
        if model_name == "Claude 4 Sonnet":
            additional_fields["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]

        model = BedrockModel(
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
            region_name=profile["bedrock_region"],
            model_id=model_id,
            max_tokens=64000,
            stop_sequences=[STOP_SEQUENCE],  # pylint: disable=possibly-used-before-assignment
            temperature=1,
            additional_request_fields=additional_fields,
        )
    else:
        model = BedrockModel(
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
            region_name=profile["bedrock_region"],
            model_id=model_id,
            max_tokens=maxOutputTokens,
            stop_sequences=[STOP_SEQUENCE],
            temperature=0.1,
            top_p=0.9,
            additional_request_fields={"thinking": {"type": "disabled"}},
        )
    return model


# Remove global conversation manager - will be session-specific
# conversation_manager = SlidingWindowConversationManager(window_size=10,)


cv_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command=".venv/bin/python",
            args=["-m", "application.aws_cv_mcp_server.server"],
            env={
                **os.environ,  # Pass all current environment variables
                "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
                "BUCKET_NAME": os.environ.get("BUCKET_NAME", "your-bucket-name"),
                # "PROFILE": os.environ.get("PROFILE"),  # Explicitly pass AWS profile
                "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
                "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY")
            }
        )
    )
)

# Add the image-opensearch MCP client with robust path handling
def _get_image_opensearch_client():
    """Create image-opensearch MCP client with robust path detection"""
    # Get the current working directory and application directory
    current_dir = Path.cwd()
    app_dir = current_dir / "application"
    server_dir = app_dir / "image-opensearch-server"

    # Try to find Python executable in virtual environment
    venv_paths = [
        server_dir / "venv" / "bin" / "python3.12",
        server_dir / "venv" / "bin" / "python3",
        server_dir / "venv" / "bin" / "python",
        server_dir / ".venv" / "bin" / "python3.12",
        server_dir / ".venv" / "bin" / "python3",
        server_dir / ".venv" / "bin" / "python",
    ]

    python_command = None
    for venv_path in venv_paths:
        if venv_path.exists():
            python_command = str(venv_path)
            break

    # Fallback to system Python if no venv found
    if python_command is None:
        python_command = sys.executable
        logger.warning(
            f"No virtual environment found for image-opensearch-server, "
            f"using system Python: {python_command}"
        )
    else:
        logger.info(f"Using image-opensearch-server Python: {python_command}")

    # Verify server directory exists
    if not server_dir.exists():
        logger.error(f"Image-opensearch server directory not found: {server_dir}")
        raise FileNotFoundError(f"Image-opensearch server directory not found: {server_dir}")

    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command=python_command,
                args=["-m", "src.server"],
                cwd=str(server_dir),
                env={
                    **os.environ,
                    "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
                    "OPENSEARCH_ENDPOINT": os.environ.get("OPENSEARCH_ENDPOINT"),
                    "PYTHONPATH": f"{server_dir}:{os.environ.get('PYTHONPATH', '')}",
                    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
                    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY")
                }
            )
        )
    )

# Initialize the image-opensearch MCP client
try:
    image_opensearch_mcp_client = _get_image_opensearch_client()
    logger.info("Image-opensearch MCP client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize image-opensearch MCP client: {e}")
    image_opensearch_mcp_client = None


#########################################################
# =========================================================================
# CORE AGENT IMPLEMENTATIONS - Framework Extension Points
# =========================================================================

def cv_agent_impl(query: str, active_client) -> str:
    """
    🎯 CV SPECIALIST AGENT (Single-Agent Pattern)

    This function implements the single-agent pattern for computer vision tasks.
    It connects a specialized CV agent directly to MCP tools for immediate processing.

    🏗️ FRAMEWORK EXTENSION PATTERN:
    1. Import agent prompt: from prompts.agent_name import SYSTEM_PROMPT
    2. Get MCP tools: cv_tools = get_tools_from_client(active_client)
    3. Create agent: Agent(model=model, system_prompt=prompt, tools=tools)
    4. Execute: agent.invoke(query)

    📋 USE CASES:
    - Single-purpose CV applications
    - Direct tool usage scenarios
    - Simple deployment patterns
    - When orchestration is not needed

    🔄 INPUT/OUTPUT:
    - Input: Natural language query describing CV task
    - Context: Available files passed via active_client context
    - Output: Success/error message (tools handle display automatically)

    Args:
        query: The CV task query (e.g., "Crop all people from image.jpg")
        active_client: Active MCP client session providing tool access

    Returns:
        Success message or error description

    Example Usage:
        result = cv_agent_impl("Remove background from photo.jpg", client)
    """
    # Validate that active_client is provided and valid
    logger.debug(f"cv_agent_impl query: {query}")
    if active_client is None:
        error_msg = "Error: Active CV client session is required but not provided"
        logger.error(error_msg)
        return error_msg

    try:
        # 🛠️ GET MCP TOOLS - Framework Extension Point
        # This retrieves all available tools from the MCP server
        # New tools added to aws_cv_mcp_server/cv_tools.py automatically appear here
        cv_tools = active_client.list_tools_sync()
        if not cv_tools:
            error_msg = (
                "Error: CV MCP server has no available tools. "
                "Check aws_cv_mcp_server/server.py tool registrations."
            )
            logger.error(error_msg)
            return error_msg

        #logger.info(f"Available CV tools: {[tool.name for tool in cv_tools]}")

        # 🎯 CREATE SPECIALIZED AGENT - Framework Core Pattern
        # Agent = Model + System Prompt + Tools
        system_prompt = CV_SYSTEM_PROMPT

        model = get_model()
        logger.info(f"\nSending the following query to: {model.get_config()['model_id']}:\n'{query}'")



        cv_agent = Agent(model=model, system_prompt=system_prompt, tools=cv_tools)

        response = cv_agent(query)
        return str(response)
    except Exception as e:
        error_msg = f"Error in cv agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


def image_opensearch_agent_impl(query: str, active_client) -> str:
    """
    🔍 IMAGE OPENSEARCH SPECIALIST AGENT (Single-Agent Pattern)

    This function implements the single-agent pattern for image search and database tasks.
    It connects a specialized OpenSearch agent to MCP tools for image operations.

    📋 USE CASES:
    - Image description generation
    - Multimodal embedding creation
    - OpenSearch indexing operations
    - Image similarity searches
    - URL-based image processing

    🔄 INPUT/OUTPUT:
    - Input: Natural language query for image search/database operations
    - Context: Image URLs or database queries via active_client
    - Output: Search results, descriptions, or operation confirmations

    Args:
        query: The image search/database task query
        active_client: Active MCP client session providing OpenSearch tools

    Returns:
        Success message, search results, or error description
    """
    logger.debug(f"image_opensearch_agent_impl query: {query}")
    if active_client is None:
        error_msg = "Error: Active Image OpenSearch client session is required but not provided"
        logger.error(error_msg)
        return error_msg

    try:
        # Get OpenSearch-specific tools
        opensearch_tools = active_client.list_tools_sync()
        if not opensearch_tools:
            error_msg = (
                "Error: Image OpenSearch MCP server has no available tools. "
                "Check image-opensearch-server configuration."
            )
            logger.error(error_msg)
            return error_msg

        # Create specialized system prompt for OpenSearch operations
        # Using CV prompt for now, but could be specialized later
        system_prompt = CV_SYSTEM_PROMPT

        model = get_model()
        logger.info(f"\nSending OpenSearch query to: {model.get_config()['model_id']}:\n'{query}'")

        opensearch_agent = Agent(model=model, system_prompt=system_prompt, tools=opensearch_tools)

        response = opensearch_agent(query)
        return str(response)
    except Exception as e:
        error_msg = f"Error in image opensearch agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


#########################################################
# Dynamic Tool Creation with MCP Client Closures
#########################################################

def create_specialist_tools_with_clients(cv_client, image_opensearch_client):
    """
    🛠️ DYNAMIC TOOL CREATION - Thread-Safe MCP Client Distribution

    Creates agent tools with MCP clients captured in closures, eliminating the need
    for session state and making tools thread-safe for background execution.

    🏗️ ARCHITECTURE BENEFITS:
    - No nested MCP connections (single connection per server type)
    - No Streamlit session state dependency (thread-safe)
    - Clean separation of concerns (tools get clients via closures)
    - Framework compatible (works with Strands background threads)

    Args:
        cv_client: Active CV MCP client connection
        image_opensearch_client: Active Image OpenSearch MCP client connection

    Returns:
        List of specialist agent tools with embedded client connections
    """

    @tool
    def cv_agent(query: str) -> str:
        """
        Specialized agent for performing computer vision tasks.
        Has tools to crop describe images, detect labels, bounding boxes, crop images and more.

        Args:
            query: The cv task query

        Returns:
            Success message or error message
        """
        # Use client captured in closure (thread-safe, no session state dependency)
        try:
            if cv_client is None:
                return "Error: CV client connection not available. Please ensure MCP server is running."
            return cv_agent_impl(query, cv_client)
        except Exception as e:
            logger.error(f"CV agent execution failed: {str(e)}")
            return f"Error: CV agent execution failed: {str(e)}"

    @tool
    def image_opensearch_agent(query: str) -> str:
        """
        Specialized agent for working with image descriptions, embeddings and OpenSearch operations.
        Has tools to generate image descriptions, create embeddings, ingest images into OpenSearch,
        and search for similar images using text or image queries.

        Args:
            query: The image search task query (e.g., "Describe this image", "Search for similar images")

        Returns:
            Success message or error message
        """
        # Use client captured in closure (thread-safe, no session state dependency)
        try:
            if image_opensearch_client is None:
                return "Error: Image OpenSearch client connection not available. Please ensure MCP server is running."
            return image_opensearch_agent_impl(query, image_opensearch_client)
        except Exception as e:
            logger.error(f"Image OpenSearch agent execution failed: {str(e)}")
            return f"Error: Image OpenSearch agent execution failed: {str(e)}"

    # Return the dynamically created tools
    return [cv_agent, image_opensearch_agent]

def get_session_images():
    """Get the pending and current operation images from the session state.
    
    Returns:
        tuple: A tuple containing (pending_images, current_operation_images)
    """
    import streamlit as st  # pylint: disable=import-outside-toplevel
    if 'pending_images' not in st.session_state:
        st.session_state.pending_images = []
    if 'current_operation_images' not in st.session_state:
        st.session_state.current_operation_images = []
    return st.session_state.pending_images, st.session_state.current_operation_images

def get_conversation_manager():
    """Get the conversation manager from the session state.
    
    Returns:
        SlidingWindowConversationManager: The conversation manager instance
    """
    import streamlit as st  # pylint: disable=import-outside-toplevel
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = SlidingWindowConversationManager(window_size=10)
    return st.session_state.conversation_manager

#########################################################
# Agent Tool Wrappers for Orchestrator
#########################################################
@tool
def ui_show_image(image_source: str, caption: str = "") -> str:
    """
    Display an image in the Streamlit UI from either S3 or URL.

    Args:
        image_source: Either a URL (http://... or https://...) or S3 key/path
                     (e.g., "mcp/filename.jpg" or just "filename.jpg")
        caption: Optional caption for the image

    Returns:
        Success message confirming image will be displayed
    """
    try:
        logger.info(f"ui_show_image called with image source: {image_source}")

        # Detect if it's a URL or S3 key
        if image_source.startswith(("http://", "https://")):
            # It's a URL - use as-is
            image_data = {
                "image_url": image_source,
                "caption": caption or f"Image: {image_source}"
            }
            logger.info(f"Detected URL image: {image_source}")
        else:
            # It's an S3 key - ensure it has the mcp/ prefix
            if not image_source.startswith("mcp/"):
                s3_key = f"mcp/{image_source}"
            else:
                s3_key = image_source

            image_data = {
                "s3_key": s3_key,
                "caption": caption or f"Image: {s3_key}"
            }
            logger.info(f"Detected S3 key: {s3_key}")

        # Store image data in thread-safe global storage
        # This allows background agent threads to communicate with main UI thread
        add_image_to_global_storage(image_data)

        success_msg = f"Successfully prepared image for display: {image_source}"
        logger.info(success_msg)

        return success_msg

    except Exception as e:
        error_msg = f"Error preparing image display for source {image_source}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def ui_show_images(image_sources: List[str], captions: List[str] = None) -> str:
    """
    Display multiple images in a grid layout from either S3 or URLs.

    Args:
        image_sources: List of image sources - either URLs (http://... or https://...) or S3 keys/filenames
        captions: Optional list of captions for each image

    Returns:
        Status message
    """
    try:
        logger.info(f"ui_show_images called with {len(image_sources)} image sources: {image_sources}")

        # Get bucket name from environment
        bucket_name = os.environ.get("BUCKET_NAME", "your-bucket-name")

        for i, image_source in enumerate(image_sources):
            logger.info(f"Processing image {i+1}/{len(image_sources)}: {image_source}")

            try:
                caption = captions[i] if captions and i < len(captions) else f"Image: {image_source}"

                # Detect if it's a URL or S3 key
                if image_source.startswith(("http://", "https://")):
                    # It's a URL - use as-is
                    image_data = {
                        "image_url": image_source,
                        "caption": caption
                    }
                    logger.info(f"Detected URL image {i+1}: {image_source}")
                else:
                    # It's an S3 key - ensure it has the mcp/ prefix
                    if not image_source.startswith("mcp/"):
                        full_s3_key = f"mcp/{image_source}"
                    else:
                        full_s3_key = image_source

                    image_data = {
                        "s3_key": full_s3_key,
                        "caption": caption
                    }
                    logger.info(f"Detected S3 key {i+1}: s3://{bucket_name}/{full_s3_key}")

                # Store each image in thread-safe global storage
                add_image_to_global_storage(image_data)

                logger.info(f"Successfully prepared image {i+1}: {image_source}")

            except Exception as e:
                error_msg = f"Error preparing image {image_source}: {str(e)}"
                logger.error(error_msg)
                # Add an error placeholder
                error_image_data = {
                    "s3_key": "",
                    "caption": f"Error: {image_source}",
                    "error": True
                }
                add_image_to_global_storage(error_image_data)

        success_msg = f"Successfully prepared {len(image_sources)} images for batch display"
        logger.info(success_msg)

        return success_msg

    except Exception as e:
        error_msg = f"Error in batch image display: {str(e)}"
        logger.error(error_msg)
        return error_msg



def create_interaction_agent(history_mode, tools, st=None):  # pylint: disable=unused-argument
    """
    🤝 INTERACTION AGENT (Multi-Agent Orchestration Pattern)

    This function implements the multi-agent orchestration pattern where an
    interaction agent coordinates between multiple specialized agents.

    🏗️ FRAMEWORK EXTENSION PATTERN:
    1. Create specialist agents as @tool decorated functions
    2. Add specialist tools to the tools list
    3. Update interaction_agent.py prompt to reference new specialists
    4. Follow delegation pattern: analyze → delegate → coordinate → respond

    📋 USE CASES:
    - Complex workflows requiring multiple specialists
    - Cross-domain tasks (CV + NLP + reporting)
    - Sophisticated orchestration scenarios
    - When agent coordination is needed

    🔄 ORCHESTRATION FLOW:
    Input: User query → Analyze: Determine specialists needed →
    Delegate: Call specialist agents → Coordinate: Synthesize results →
    Output: Unified response

    🛠️ SPECIALIST INTEGRATION:
    - cv_agent tool: Delegates CV tasks to CV specialist
    - Future: nlp_agent, report_agent, data_agent tools
    - Each specialist handles domain-specific tool usage
    - Interaction agent focuses on coordination, not direct tool usage

    Args:
        history_mode: "Enable"/"Disable" conversation memory
        tools: List of specialist agent tools (e.g., [cv_agent_tool])

    Returns:
        Configured interaction agent ready for orchestration

    Example Usage:
        agent = create_interaction_agent("Enable", [cv_agent_tool, report_agent_tool])
        response = agent.invoke("Analyze images and generate report")
    """
    # Get orchestrator system prompt - simple direct import
    from prompts.interaction_agent import SYSTEM_PROMPT as INTERACTION_SYSTEM_PROMPT  # pylint: disable=import-outside-toplevel
    system = INTERACTION_SYSTEM_PROMPT

    model = get_model()
    logger.info(f"\nCreating orchestrator agent with model: {model.get_config()['model_id']}")

    try:
        # Specialist agent tools are passed in - these enable delegation
        # Framework Extension: Add new specialist tools to the tools list

        # Always use conversation manager for session memory
        logger.info("Using conversation manager for session memory")
        conv_manager = get_conversation_manager() if st else SlidingWindowConversationManager(window_size=10)

        # Create agent
        orchestrator = Agent(
            model=model,
            system_prompt=system,
            tools=tools,
            conversation_manager=conv_manager,
        )

        # Initialize agent with Streamlit conversation history
        if st and hasattr(st.session_state, 'messages') and st.session_state.messages:
            streamlit_messages = st.session_state.messages
            logger.info(f"Initializing agent with {len(streamlit_messages)} messages from Streamlit")

            # Convert Streamlit messages to Strands message format
            # SEC M3 — Conversation history is re-injected without
            # sanitisation. The SlidingWindowConversationManager
            # (window_size=10) limits how far back content persists.
            # For production, tag and strip assistant messages that
            # originated from external content (image descriptions,
            # URL analysis) to mitigate stored prompt injection.
            strands_messages = []
            for msg in streamlit_messages:
                role = msg["role"]
                content = msg["content"]

                # Convert to Strands message format
                strands_message = {
                    "role": role,
                    "content": [{"text": content}]
                }
                strands_messages.append(strands_message)

            # Set the agent's message history
            orchestrator.messages = strands_messages
            logger.info(f"Agent initialized with {len(strands_messages)} conversation messages")
        else:
            logger.info("No Streamlit messages found, starting with empty conversation")

        return orchestrator
    except Exception as e:
        logger.error(f"Error initializing orchestrator agent: {e}")
        # If error occurs, substitute with a basic agent
        return Agent(
            model=model,
            system_prompt=system,
            tools=tools if "tools" in locals() else [],
        )




def run_multi_agent_system(question, history_mode, st):
    """Run the multi-agent system to process the user's question.
    
    Args:
        question (str): The user's question
        history_mode (str): The history mode to use
        st: The Streamlit instance
        
    Returns:
        str: The full response from the multi-agent system
    """
    # Clear global image storage at the start of each new request
    clear_global_image_storage()

    message_placeholder = st.empty()
    full_response = ""

    async def process_streaming_response():
        nonlocal full_response
        try:
            # Validate MCP clients before opening sessions
            if image_opensearch_mcp_client is None:
                full_response = (
                    "Error: Image OpenSearch MCP client is not available. "
                    "Please check server configuration."
                )
                message_placeholder.markdown(full_response)
                return full_response, []

            # Open all client sessions at once and manage them
            with cv_mcp_client as cv_client, image_opensearch_mcp_client as image_client:

                logger.info("Creating thread-safe specialist tools with MCP client closures")

                # 🛠️ DYNAMIC TOOL CREATION - Thread-Safe MCP Client Distribution
                # Create specialist tools using closures to capture MCP clients
                # This eliminates session state dependency and threading issues
                specialist_tools_dynamic = create_specialist_tools_with_clients(cv_client, image_client)

                # Add UI tools to the specialist tools list
                specialist_tools = specialist_tools_dynamic + [ui_show_image, ui_show_images]

                logger.info(
                    f"Created {len(specialist_tools)} specialist tools: "
                    f"{[tool.__name__ for tool in specialist_tools]}"
                )

                current_orchestrator = create_interaction_agent(history_mode, specialist_tools, st)

                # Stream the orchestrator response
                agent_stream = current_orchestrator.stream_async(question)
                async for event in agent_stream:
                    if "data" in event:
                        full_response += event["data"]
                        message_placeholder.markdown(full_response)

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            message_placeholder.markdown(
                "Sorry, an error occurred while generating the response."
            )
            logger.error(traceback.format_exc())  # Detailed error logging

    asyncio.run(process_streaming_response())

    # Retrieve images from thread-safe global storage
    # This is where background agent threads communicate image data to the main thread
    images_to_add = get_images_from_global_storage()

    logger.info(f"Returning {len(images_to_add)} images with response")
    return full_response, images_to_add


def clear_chat_history():
    """Clear chat history and reset conversation manager"""
    import streamlit as st  # pylint: disable=import-outside-toplevel

    # Reset session-specific state
    st.session_state.conversation_manager = SlidingWindowConversationManager(window_size=10)
    st.session_state.pending_images = []
    st.session_state.current_operation_images = []
    if hasattr(st.session_state, 'current_images'):
        st.session_state.current_images = []

    logger.info("Chat history and image caches cleared")
