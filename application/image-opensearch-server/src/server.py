
"""Image OpenSearch MCP Server implementation."""

import os
import json
import base64
import ipaddress
import logging
import re
import socket
import time
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import signal

# Handle pipe errors gracefully
from signal import signal, SIGPIPE, SIG_DFL

import httpx
from mcp.server.fastmcp import FastMCP

# Ignore SIGPIPE errors
signal(SIGPIPE, SIG_DFL)

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("image_opensearch_server")


def validate_image_url(url: str) -> tuple:
    """Validate image URL to prevent SSRF attacks (CWE-918).

    Blocks private IPs, loopback, link-local, and AWS metadata service.

    Args:
        url: The URL to validate.

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ('http', 'https'):
            return False, f"Invalid scheme: {parsed.scheme}"

        hostname = parsed.hostname
        if not hostname:
            return False, "No hostname in URL"

        # Resolve hostname to IP and validate
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        # Block private, loopback, link-local, reserved, and AWS metadata service
        if (ip_obj.is_private or ip_obj.is_loopback
                or ip_obj.is_link_local or ip_obj.is_reserved
                or ip == "169.254.169.254"):
            return False, "Access to private/internal addresses is not allowed"

        return True, None

    except socket.gaierror:
        return False, f"Could not resolve hostname: {hostname}"  # pylint: disable=used-before-assignment
    except Exception as e:
        return False, str(e)

# Import the context manager with fallback for MCP server process
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from application.image_context_manager import context_manager
except ImportError:
    # Create a dummy context manager for MCP server process
    class DummyContextManager:
        """A dummy context manager for when the real one is not available."""
        def register_index(self, index_name):
            """Register an index with the context manager.

            Args:
                index_name (str): The name of the index to register
            """

        def record_operation(self, op_type, details):
            """Record an operation in the context manager.

            Args:
                op_type (str): The type of operation
                details (dict): Details about the operation
            """

        def record_search_results(self, query_type, query, results):
            """Record search results in the context manager.

            Args:
                query_type (str): The type of query
                query (str): The query that was executed
                results (dict): The search results
            """

        def update_entity_count(self, index_name, entity_type, count):
            """Update the entity count in the context manager.

            Args:
                index_name (str): The name of the index
                entity_type (str): The type of entity
                count (int): The new count
            """

        def get_default_index(self):
            """Get the default index.

            Returns:
                str or None: The default index name, or None if not set
            """
            return None

        def get_entity_count(self, index_name, entity_type):  # pylint: disable=unused-argument
            """Get the entity count for a specific index and entity type.

            Args:
                index_name (str): The name of the index
                entity_type (str): The type of entity

            Returns:
                int or None: The entity count, or None if not found
            """
            return None
    context_manager = DummyContextManager()

# Handle imports correctly whether run as script or module
try:
    # When imported as a module
    from .util import get_bedrock_client, get_opensearch_client
except ImportError:
    # When run directly as a script
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from src.util import get_bedrock_client, get_opensearch_client

mcp = FastMCP(
    'image-opensearch-server',
    instructions=(
        'Generate image descriptions, create embeddings, and ingest/query images '
        'in OpenSearch collections using AWS Bedrock and Amazon Titan models'
    ),
    dependencies=[
        'boto3',
        'opensearch-py',
        'httpx',
        'Pillow',
        'pydantic',
        'loguru',
        'requests-aws4auth'
    ],
)


class RateLimiter:
    """Token-bucket rate limiter for expensive operations (H4)."""

    def __init__(self, max_calls, period_seconds):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed per period.
            period_seconds: Time window in seconds.
        """
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls = []

    def allow(self):
        """Check if a call is allowed under the rate limit.

        Returns:
            bool: True if the call is allowed.
        """
        now = time.time()
        self._calls = [t for t in self._calls
                       if now - t < self.period]
        if len(self._calls) >= self.max_calls:
            return False
        self._calls.append(now)
        return True


# SEC: H4 — Rate limiter for expensive bulk ingest
_bulk_ingest_limiter = RateLimiter(max_calls=3, period_seconds=60)


def is_base64_encoded(s: str) -> bool:
    """Check if string is base64 encoded.

    Args:
        s (str): String to check

    Returns:
        bool: True if string is base64 encoded, False otherwise
    """
    if len(s) % 4 != 0:
        return False
    return bool(re.match('^[A-Za-z0-9+/]+={0,2}$', s))


async def generate_image_description_internal(
    bedrock_client,
    image_url: str,
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
) -> str:
    """Generate image description using AWS Bedrock Claude model.

    Args:
        bedrock_client: The Bedrock client to use
        image_url (str): URL of the image to describe
        model_id (str): Bedrock model ID for image description

    Returns:
        str: The image description
    """
    logger.debug(f"Attempting to use model: {model_id}")

    # SEC: Validate URL to prevent SSRF (CWE-918)
    is_valid, error_msg = validate_image_url(image_url)
    if not is_valid:
        raise ValueError(f"URL validation failed: {error_msg}")

    # Download and encode image
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            follow_redirects=False,  # SEC: Prevent redirect-based SSRF
        ) as client:
            logger.debug(f"Downloading image from {image_url}")
            response = await client.get(image_url)

            # Resize image if too large
            from PIL import Image  # pylint: disable=import-outside-toplevel
            from io import BytesIO  # pylint: disable=import-outside-toplevel
            image = Image.open(BytesIO(response.content))
            if image.width > 500 or image.height > 500:
                image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                image_data = base64.b64encode(response.content).decode("utf-8")

            logger.debug(f"Image downloaded and encoded, size: {len(image_data)} chars")
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise

    system = (
        "You interpret the image and tell the user what it is about concisely, "
        "Do not generate preamble such as 'This image shows/depicts..' in the answer."
    )

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 128,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in these images?"
                    }
                ]
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "top_k": 50,
        "stop_sequences": ["Human:"]
    }

    try:
        logger.debug(f"Calling Bedrock with model_id={model_id}")
        response = bedrock_client.invoke_model(
            body=json.dumps(request_body),
            modelId=model_id
        )
        logger.debug("Bedrock call successful")

        response_body = json.loads(response.get('body').read())
        return response_body.get('content')[0]['text']
    except Exception as e:
        logger.error(f"Error calling Bedrock: {str(e)}", exc_info=True)
        raise


async def generate_multimodal_embedding_internal(
    bedrock_client,
    image: str,
    image_description: str
) -> List[float]:
    """Generate multimodal embedding using Amazon Titan.

    Args:
        bedrock_client: The Bedrock client to use
        image (str): Image URL or base64 encoded image data
        image_description (str): Text description of the image

    Returns:
        List[float]: The embedding vector
    """
    if is_base64_encoded(image):
        image_data = image
    else:
        # SEC: Validate URL to prevent SSRF (CWE-918)
        is_valid, error_msg = validate_image_url(image)
        if not is_valid:
            raise ValueError(f"URL validation failed: {error_msg}")

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            follow_redirects=False,
        ) as client:
            response = await client.get(image)
            # image_data = base64.b64encode(response.content).decode("utf-8")

            from PIL import Image  # pylint: disable=import-outside-toplevel
            from io import BytesIO  # pylint: disable=import-outside-toplevel
            image = Image.open(BytesIO(response.content))
            if image.width > 500 or image.height > 500:
                image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                image_data = base64.b64encode(response.content).decode("utf-8")




    body = json.dumps({
        "inputImage": image_data,
        "inputText": image_description
    })

    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    return response_body['embedding']


async def generate_text_embedding_internal(bedrock_client, text: str) -> List[float]:
    """Generate text embedding using Amazon Titan.

    Args:
        bedrock_client: The Bedrock client to use
        text (str): Text to generate embedding for

    Returns:
        List[float]: The embedding vector
    """
    body = json.dumps({
        "inputText": text
    })

    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    return response_body['embedding']


async def generate_image_embedding_internal(bedrock_client, image_url: str) -> List[float]:
    """Generate image embedding using Amazon Titan.

    Args:
        bedrock_client: The Bedrock client to use
        image_url (str): URL of the image to generate embedding for

    Returns:
        List[float]: The embedding vector
    """
    # SEC: Validate URL to prevent SSRF (CWE-918)
    is_valid, error_msg = validate_image_url(image_url)
    if not is_valid:
        raise ValueError(f"URL validation failed: {error_msg}")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(5.0, connect=2.0),
        follow_redirects=False,
    ) as client:
        response = await client.get(image_url)
        image_data = base64.b64encode(response.content).decode("utf-8")

    body = json.dumps({
        "inputImage": image_data
    })

    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    return response_body['embedding']


@mcp.tool(name='generate_image_description')
async def generate_image_description_tool(
    image_url: str,
    model_id: Optional[str] = "anthropic.claude-3-sonnet-20240229-v1:0",
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a concise description of an image using AWS Bedrock Claude model.

    Parameters:
        image_url (str): URL of the image to describe.
        model_id (str, optional): Bedrock model ID for image description.
        region (str, optional): AWS region to use.

    Returns:
        Dict containing the image URL, description, and model ID used.
    """
    # List of models to try in order
    models_to_try = [
        model_id,  # First try the requested model
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0"
    ]

    last_error = None
    bedrock_client = get_bedrock_client(region)

    # Try each model in succession until one works
    for current_model in models_to_try:
        try:
            logger.debug(f"Attempting with model: {current_model}")
            description = await generate_image_description_internal(bedrock_client, image_url, current_model)

            return {
                'image_url': image_url,
                'description': description,
                'model_id': current_model,  # Return which model actually worked
                'region': region or os.environ.get('AWS_REGION', 'us-east-1')
            }
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Failed with model {current_model}: {last_error}")
            continue  # Try next model

    # If we get here, all models failed
    return {
        'error': f"All models failed. Last error: {last_error}",
        'image_url': image_url
    }


@mcp.tool(name='generate_multimodal_embedding')
async def generate_multimodal_embedding_tool(
    image: str,
    image_description: str,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate multimodal embedding for image and text using Amazon Titan.

    Parameters:
        image (str): Image URL or base64 encoded image data.
        image_description (str): Text description of the image.
        region (str, optional): AWS region to use.

    Returns:
        Dict containing the embedding and metadata.
    """
    try:
        bedrock_client = get_bedrock_client(region)
        embedding = await generate_multimodal_embedding_internal(bedrock_client, image, image_description)

        return {
            'embedding_length': len(embedding),
            'embedding': embedding,
            'image': image[:100] + '...' if len(image) > 100 else image,  # Truncate for display
            'image_description': image_description,
            'region': region or os.environ.get('AWS_REGION', 'us-east-1')
        }
    except Exception as e:
        return {'error': str(e), 'image_description': image_description}


@mcp.tool(name='ingest_image_to_opensearch')
async def ingest_image_to_opensearch_tool(
    image_url: str,
    index_name: str = "imgs-vector-index-test",
    model_id: Optional[str] = "anthropic.claude-3-haiku-20240307-v1:0",
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate description and embedding for an image and ingest into OpenSearch."""
    try:
        logger.info(f"Starting image ingestion for {image_url} to index {index_name}")

        # Register the index with the context manager
        context_manager.register_index(index_name)

        # SEC: Credential logging removed to prevent sensitive info disclosure (LLM06)
        logger.debug(f"AWS_REGION: {os.environ.get('AWS_REGION', 'Not set')}")

        # Get caller identity for debugging
        try:
            import boto3  # pylint: disable=import-outside-toplevel
            session = boto3.Session(
                region_name=os.environ.get("AWS_REGION"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            )

            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            logger.debug(f"Current identity: {identity['Arn']}")
        except Exception as e:
            logger.warning(f"Failed to get identity: {e}")

        # Continue with normal flow
        bedrock_client = get_bedrock_client(region)
        if not bedrock_client:
            return {'error': "Failed to create Bedrock client", 'image_url': image_url, 'index_name': index_name}

        opensearch_client = get_opensearch_client(region)
        if not opensearch_client:
            return {'error': "Failed to create OpenSearch client", 'image_url': image_url, 'index_name': index_name}

        # Check if index exists, create it if it doesn't
        if not opensearch_client.indices.exists(index=index_name):
            logger.info(f"Creating index {index_name}")
            mapping = {
                "settings": {"index.knn": "true"},
                "mappings": {
                    "properties": {
                        "image_vector": {
                            "type": "knn_vector",
                            "dimension": 1024  # Titan embedding dimension
                        },
                        "photo_image_url": {"type": "keyword"},
                        "photo_description": {"type": "text"},
                        "timestamp": {"type": "date"}
                    }
                }
            }
            opensearch_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created index {index_name} successfully")

        # Generate description
        logger.info("Generating image description...")
        description = await generate_image_description_internal(bedrock_client, image_url, model_id)
        logger.debug(f"Generated description: {description[:100]}...")

        # Generate embedding
        logger.info("Generating embedding...")
        embedding = await generate_multimodal_embedding_internal(bedrock_client, image_url, description)
        logger.debug(f"Generated embedding of length {len(embedding)}")

        # Create document
        document = {
            "image_vector": embedding,
            "photo_image_url": image_url,
            "photo_description": description,
            "timestamp": int(time.time() * 1000)
        }

        # Index document
        logger.info(f"Indexing document to {index_name}...")
        response = opensearch_client.index(
            index=index_name,
            body=document
        )

        result = {
            'success': True,
            'document_id': response["_id"],
            'index': index_name,
            'description': description,
            'embedding_length': len(embedding),
            'image_url': image_url,
            'region': region or os.environ.get('AWS_REGION', 'us-east-1')
        }

        # Record the operation in the context manager
        context_manager.record_operation('ingest', {
            'index_name': index_name,
            'image_url': image_url,
            'description': description,
            'document_id': response["_id"]
        })

        # If the description contains animal references, update the animal count
        if any(animal in description.lower() for animal in ['animal', 'cat', 'dog', 'bird', 'fish', 'pet']):
            # Get current count or default to 0
            current_count = context_manager.get_entity_count(index_name, 'animal') or 0
            # Increment the count
            context_manager.update_entity_count(index_name, 'animal', current_count + 1)

        return result
    except Exception as e:
        logger.error(f"Error in ingest_image_to_opensearch_tool: {e}", exc_info=True)
        return {'error': str(e), 'image_url': image_url, 'index_name': index_name}


@mcp.tool(name='query_images_by_text')
async def query_images_by_text_tool(
    input_text: str,
    index_name: str = None,
    topk: Optional[int] = 5,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for images in OpenSearch using text query.

    Parameters:
        input_text (str): Text query to search for similar images.
        index_name (str): OpenSearch index name (case sensitive)
        topk (int, optional): Number of top results to return (default: 5).
        region (str, optional): AWS region to use.

    Returns:
        Dict containing search results from OpenSearch.
    """
    try:
        bedrock_client = get_bedrock_client(region)
        opensearch_client = get_opensearch_client()

        # If index_name is not provided, get the default index
        if not index_name:
            index_name = context_manager.get_default_index()
            if not index_name:
                return {'error': "No index name provided and no default index available", 'query': input_text}

        # Register the index
        context_manager.register_index(index_name)

        text_embedding = await generate_text_embedding_internal(bedrock_client, input_text)

        query = {
            "size": topk,
            "_source": {"excludes": ["image_vector"]},
            "query": {
                "knn": {
                    "image_vector": {
                        "vector": text_embedding,
                        "k": 10
                    }
                }
            }
        }

        response = opensearch_client.search(
            index=index_name,
            body=query
        )

        result = {
            'query': input_text,
            'index': index_name,
            'total_results': response.get('hits', {}).get('total', {}).get('value', 0),
            'results': response.get('hits', {}).get('hits', []),
            'region': region or os.environ.get('AWS_REGION', 'us-east-1')
        }

        # Record the search results in context manager
        context_manager.record_search_results('text', input_text, result)

        # If searching for animals, update the animal count
        if any(animal in input_text.lower() for animal in ['animal', 'cat', 'dog', 'bird', 'fish', 'pet']):
            total = result['total_results']
            context_manager.update_entity_count(index_name, 'animal', total)

        return result
    except Exception as e:
        return {'error': str(e), 'query': input_text, 'index_name': index_name}


@mcp.tool(name='query_images_by_image')
async def query_images_by_image_tool(
    image_url: str,
    index_name: str,
    topk: Optional[int] = 5,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for similar images in OpenSearch using image query.

    Parameters:
        image_url (str): URL of the query image.
        index_name (str): OpenSearch index name.
        topk (int, optional): Number of top results to return (default: 5).
        region (str, optional): AWS region to use.

    Returns:
        Dict containing search results from OpenSearch.
    """
    try:
        bedrock_client = get_bedrock_client(region)
        opensearch_client = get_opensearch_client()

        image_embedding = await generate_image_embedding_internal(bedrock_client, image_url)

        query = {
            "size": topk,
            "_source": {"excludes": ["image_vector"]},
            "query": {
                "knn": {
                    "image_vector": {
                        "vector": image_embedding,
                        "k": 10
                    }
                }
            }
        }

        response = opensearch_client.search(
            index=index_name,
            body=query
        )

        return {
            'query_image_url': image_url,
            'index': index_name,
            'total_results': response.get('hits', {}).get('total', {}).get('value', 0),
            'results': response.get('hits', {}).get('hits', []),
            'region': region or os.environ.get('AWS_REGION', 'us-east-1')
        }
    except Exception as e:
        return {'error': str(e), 'image_url': image_url, 'index_name': index_name}


@mcp.tool(name='bulk_ingest_images')
async def bulk_ingest_images_tool(
    image_urls: List[str],
    index_name: str,
    model_id: Optional[str] = "anthropic.claude-3-sonnet-20240229-v1:0",
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Bulk ingest multiple images into OpenSearch with generated descriptions and embeddings.

    Parameters:
        image_urls (List[str]): Array of image URLs to ingest.
        index_name (str): OpenSearch index name.
        model_id (str, optional): Bedrock model ID for description generation.
        region (str, optional): AWS region to use.

    Returns:
        Dict containing bulk ingestion results and statistics.
    """
    try:
        # SEC: H4 — Rate limit expensive bulk ingest
        if not _bulk_ingest_limiter.allow():
            return {
                'error': 'Rate limit exceeded. Please wait before submitting another bulk ingest.',
                'image_urls_count': len(image_urls),
                'index_name': index_name,
            }

        bedrock_client = get_bedrock_client(region)
        opensearch_client = get_opensearch_client()

        # Register the index with context manager
        context_manager.register_index(index_name)

        results = []

        for image_url in image_urls:
            try:
                # Generate description
                description = await generate_image_description_internal(bedrock_client, image_url, model_id)

                # Generate embedding
                embedding = await generate_multimodal_embedding_internal(bedrock_client, image_url, description)

                # Create document
                document = {
                    "image_vector": embedding,
                    "photo_image_url": image_url,
                    "photo_description": description,
                    "timestamp": int(time.time() * 1000)
                }

                # Index document
                response = opensearch_client.index(
                    index=index_name,
                    body=document
                )

                results.append({
                    "image_url": image_url,
                    "success": True,
                    "document_id": response["_id"],
                    "description": description
                })

            except Exception as e:
                results.append({
                    "image_url": image_url,
                    "success": False,
                    "error": str(e)
                })

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        result = {
            'total_processed': len(image_urls),
            'successful': len(successful),
            'failed': len(failed),
            'results': results,
            'index': index_name,
            'model_id': model_id,
            'region': region or os.environ.get('AWS_REGION', 'us-east-1')
        }

        # Record the operation in context manager
        context_manager.record_operation('bulk_ingest', {
            'index_name': index_name,
            'image_count': len(image_urls),
            'success_count': len(successful),
            'fail_count': len(failed)
        })

        # Update animal count based on successful descriptions
        animal_count = 0
        for item in successful:
            if 'description' in item and any(
                animal in item['description'].lower()
                for animal in ['animal', 'cat', 'dog', 'bird', 'fish', 'pet']
            ):
                animal_count += 1

        if animal_count > 0:
            current_count = context_manager.get_entity_count(index_name, 'animal') or 0
            context_manager.update_entity_count(index_name, 'animal', current_count + animal_count)

        return result
    except Exception as e:
        return {'error': str(e), 'image_urls_count': len(image_urls), 'index_name': index_name}


def main():
    """Run the MCP server with stdio transport.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        logger.info("Starting MCP server with stdio transport...")
        mcp.run(transport='stdio')
    except BrokenPipeError:
        # This is normal when the client disconnects
        logger.info("Client disconnected (broken pipe)")
        return 0  # Exit with success code
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        return 1  # Exit with error code
    return 0

if __name__ == '__main__':
    main()
