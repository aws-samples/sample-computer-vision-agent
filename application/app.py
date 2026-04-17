"""Streamlit application for Computer Vision MCP Server.

This application provides a web interface for uploading and analyzing images and videos
using AWS services through the MCP (Model Context Protocol) framework.
"""

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Adapted from https://github.com/kyopark2014/strands-agent
# SPDX-License-Identifier: MIT

import logging
import sys
import os
import uuid
import boto3
import streamlit as st
from dotenv import load_dotenv

# Add the application directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import chat after setting up path
import chat  # pylint: disable=wrong-import-position

# Set up logging first so we can use it for AWS setup
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("streamlit")

# Load environment variables from .env file
load_dotenv()

session = boto3.Session(
    region_name=os.environ.get("AWS_REGION"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
)


def media_input_section():
    """
    Media input section for the Streamlit app - supports images and videos.
    """
    st.header("📸🎥 Media Upload")

    with st.container():
        uploaded_file = st.file_uploader(
            "Upload an Image or Video",
            type=["png", "jpg", "jpeg", "gif", "webp", "mp4", "avi", "mov", "mkv", "webm"],
            key="media_upload"
        )

        if uploaded_file is not None:
            # Determine if it's an image or video
            file_extension = uploaded_file.name.lower().split('.')[-1]
            is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv', 'webm']

            if is_video:
                st.video(uploaded_file)
                st.write(f"📹 Video: {uploaded_file.name}")
            else:
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

            # Generate unique filename using session ID
            file_name = f"{st.session_state.session_id}_{uploaded_file.name}"

            # Get file bytes directly from uploaded file
            file_bytes = uploaded_file.getvalue()

            if file_bytes:
                # SEC: Validate file content matches expected type (CWE-434)
                _valid_image_signatures = {
                    b'\xff\xd8\xff': 'image/jpeg',      # JPEG
                    b'\x89PNG\r\n\x1a\n': 'image/png',  # PNG
                    b'GIF87a': 'image/gif',              # GIF87a
                    b'GIF89a': 'image/gif',              # GIF89a
                    b'RIFF': 'image/webp',               # WebP (starts with RIFF)
                }
                _valid_video_signatures = {
                    b'\x00\x00\x00': 'video/',  # MP4/MOV (ftyp box)
                    b'\x1a\x45\xdf\xa3': 'video/webm',  # WebM/MKV
                }

                content_verified = False
                if is_video:
                    for sig in _valid_video_signatures:
                        if file_bytes[:len(sig)] == sig:
                            content_verified = True
                            break
                else:
                    for sig in _valid_image_signatures:
                        if file_bytes[:len(sig)] == sig:
                            content_verified = True
                            break

                if not content_verified:
                    st.error("File content does not match expected format. Upload rejected.")
                    logger.warning(f"File content validation failed for: {uploaded_file.name}")
                    return

                file_media_type = "Video" if is_video else "Image"
                st.success(f"{file_media_type} {uploaded_file.name} is ready for upload!")

                # Upload directly to S3 from memory (no temporary files needed)
                try:
                    # Use the session with the correct profile to create S3 client
                    s3_client = session.client("s3")
                    bucket_name = os.getenv("BUCKET_NAME", "your-bucket-name")
                    key = f"mcp/{file_name}"

                    # Determine content type based on file extension
                    content_type_map = {
                        # Images
                        'jpg': 'image/jpeg',
                        'jpeg': 'image/jpeg',
                        'png': 'image/png',
                        'gif': 'image/gif',
                        'webp': 'image/webp',
                        # Videos
                        'mp4': 'video/mp4',
                        'avi': 'video/x-msvideo',
                        'mov': 'video/quicktime',
                        'mkv': 'video/x-matroska',
                        'webm': 'video/webm'
                    }
                    content_type = content_type_map.get(file_extension, 'application/octet-stream')

                    # Upload directly from memory using put_object
                    s3_client.put_object(
                        Bucket=bucket_name,
                        Key=key,
                        Body=file_bytes,
                        ContentType=content_type
                    )
                    st.success(f"{file_media_type} uploaded to S3 as: {file_name} (Content-Type: {content_type})")

                    # Store uploaded media info (overwrite instead of append)
                    st.session_state.uploaded_images = [{
                        "filename": file_name,
                        "original_name": uploaded_file.name,
                        "s3_key": key,
                        "type": "video" if is_video else "image",
                        "content_type": content_type
                    }]

                except Exception as e:
                    st.error(f"Error uploading to S3: {str(e)}")
                    logger.error(f"S3 upload error: {e}")


# title
st.set_page_config(
    page_title="Computer Vision MCP Server",
    page_icon="✺◟(👁 ͜ʖ👁)◞✺",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

with st.sidebar:
    st.title("Menu")

    # model selection box
    modelName = st.selectbox(
        "🖊️ Choose your foundation model for analysis",
        (
            "Claude 4 Sonnet",
            "Claude 3.7 Sonnet",
            "Claude 3.5 Sonnet",
            "Claude 3.5 Haiku",
        ),
        index=1,
    )

    # extended thinking for Claude 4 Sonnet and Claude 3.7 Sonnet
    select_reasoning = st.checkbox(
        "Reasoning (Claude 4 Sonnet and Claude 3.7 Sonnet)", value=False
    )
    reasoningMode = (
        "Enable"
        if select_reasoning and modelName in ["Claude 4 Sonnet", "Claude 3.7 Sonnet"]
        else "Disable"
    )

    chat.update(modelName, reasoningMode)

    clear_button = st.button("Reset Conversation", key="clear")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

st.title("✺◟(👁 ͜ʖ👁)◞✺ Computer Vision & Video Analysis MCP Server")

# Add media upload section
media_input_section()

if clear_button is True:
    chat.initiate()

# Removed broken sync function - will initialize agent with messages instead

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False


def display_single_image(image_data, index):  # pylint: disable=too-many-statements
    """Helper function to display a single image with size constraints"""
    logger.info(
        f"Image {index}: type={type(image_data)}, "
        f"keys={list(image_data.keys()) if isinstance(image_data, dict) else 'N/A'}"
    )

    # CSS for smaller images that can fit side by side
    st.markdown(
        """
        <style>
        .stImage > img {
            max-width: 300px;
            max-height: 250px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if isinstance(image_data, str):
        if image_data.startswith("data:image"):
            st.image(image_data)
        elif image_data.startswith("http"):
            file_name = image_data[image_data.rfind("/") + 1 :]
            st.image(image_data, caption=file_name)
        else:
            # Assume it's an S3 key
            try:
                s3_client = session.client('s3')
                bucket_name = os.getenv('BUCKET_NAME', 'your-bucket-name')
                s3_key = image_data if image_data.startswith("mcp/") else f"mcp/{image_data}"

                s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                image_bytes = s3_response['Body'].read()
                st.image(image_bytes)
            except Exception as e:
                logger.error(f"Error displaying image from S3: {e}")
    elif isinstance(image_data, dict) and "s3_key" in image_data:
        try:
            logger.info("Displaying image from S3 key from dict")
            s3_key = image_data["s3_key"]
            logger.info(f"S3 key: {s3_key}")

            s3_client = session.client('s3')
            bucket_name = os.getenv('BUCKET_NAME', 'your-bucket-name')
            logger.info(f"Using bucket name: {bucket_name}")
            full_s3_key = s3_key if s3_key.startswith("mcp/") else f"mcp/{s3_key}"
            logger.info(f"Full S3 key: {full_s3_key}")

            s3_response = s3_client.get_object(Bucket=bucket_name, Key=full_s3_key)
            image_bytes = s3_response['Body'].read()
            logger.info(f"Downloaded image bytes length: {len(image_bytes)}")

            caption = image_data.get("caption", "")
            logger.info(f"About to call st.image with caption: '{caption}'")

            import io  # pylint: disable=import-outside-toplevel
            image_io = io.BytesIO(image_bytes)
            st.image(image_io, caption=caption)
            logger.info("st.image called successfully with BytesIO")

        except Exception as e:
            logger.error(f"Error displaying S3 image: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback  # pylint: disable=import-outside-toplevel
            logger.error(f"Full traceback: {traceback.format_exc()}")
    elif isinstance(image_data, dict) and "image_url" in image_data:
        try:
            logger.info("Displaying image from URL from dict")
            image_url = image_data["image_url"]
            caption = image_data.get("caption", "")
            logger.info(f"URL: {image_url}, Caption: '{caption}'")

            st.image(image_url, caption=caption)
            logger.info("st.image called successfully with URL")

        except Exception as e:
            logger.error(f"Error displaying URL image: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback  # pylint: disable=import-outside-toplevel
            logger.error(f"Full traceback: {traceback.format_exc()}")
    else:
        logger.warning(f"Unknown image data format: {type(image_data)}")


def display_chat_messages():
    """Print message history
    @returns None
    """
    logger.info("=== DISPLAY_CHAT_MESSAGES CALLED ===")
    logger.info(f"Total messages in session: {len(st.session_state.messages)}")

    for msg_idx, message in enumerate(st.session_state.messages):
        logger.info(f"Message {msg_idx}: role={message['role']}, has_images={'images' in message}")
        if 'images' in message:
            logger.info(f"  Message {msg_idx} has {len(message['images'])} images")

        with st.chat_message(message["role"]):
            if "images" in message:
                logger.info(f"Processing {len(message['images'])} images in message")

                # Display images side by side using columns
                if len(message["images"]) == 1:
                    # Single image - use full width but constrained size
                    display_single_image(message["images"][0], 0)
                elif len(message["images"]) == 2:
                    # Two images side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        display_single_image(message["images"][0], 0)
                    with col2:
                        display_single_image(message["images"][1], 1)
                elif len(message["images"]) == 3:
                    # Three images: two on top, one below
                    col1, col2 = st.columns(2)
                    with col1:
                        display_single_image(message["images"][0], 0)
                    with col2:
                        display_single_image(message["images"][1], 1)
                    display_single_image(message["images"][2], 2)
                else:
                    # Four or more images: display in pairs
                    for pair_idx in range(0, len(message["images"]), 2):
                        if pair_idx + 1 < len(message["images"]):
                            col1, col2 = st.columns(2)
                            with col1:
                                display_single_image(message["images"][pair_idx], pair_idx)
                            with col2:
                                display_single_image(message["images"][pair_idx + 1], pair_idx + 1)
                        else:
                            display_single_image(message["images"][pair_idx], pair_idx)
            st.markdown(message["content"])




display_chat_messages()


# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = (
            "Hi! I can help you analyze images and videos. "
            "Upload an image to crop objects, detect labels, "
            "or upload a video for detailed analysis!"
        )
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    # Clear all session state for conversation
    st.session_state.messages = []
    st.session_state.greetings = False

    # Clear uploaded images
    st.session_state.uploaded_images = []

    # Clear chat history and conversation manager
    chat.clear_chat_history()

    logger.info("Reset conversation: cleared all session state")
    st.rerun()

# SEC C1 — No authentication is configured. This is acceptable for
# local development and demos. For production, add an auth layer:
# ALB + Cognito, streamlit-authenticator, or an env-var password gate.

# Always show the chat input
if prompt := st.chat_input("Enter your message."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    # Add context about uploaded media to the prompt ONLY if relevant
    enhanced_prompt = prompt

    # Check if the user is actually referring to uploaded files in their query
    is_relevant_to_uploads = False
    if st.session_state.uploaded_images:
        # Look for keywords that suggest the user wants to work with uploaded files
        upload_keywords = ["upload", "uploaded", "file", "my image", "my video", "this image", "this video"]

        # Check if user mentions any uploaded filename
        for media in st.session_state.uploaded_images:
            filename_base = media['original_name'].lower()
            if filename_base in prompt.lower() or media['filename'].lower() in prompt.lower():
                is_relevant_to_uploads = True
                break

        # Check for upload-related keywords only if no specific filename mentioned
        if not is_relevant_to_uploads:
            for keyword in upload_keywords:
                if keyword in prompt.lower():
                    is_relevant_to_uploads = True
                    break

        # Only add uploaded media context if it's relevant to the query
        if is_relevant_to_uploads:
            uploaded_files = []
            for media in st.session_state.uploaded_images:
                media_type = media.get("type", "image")
                uploaded_files.append(f"{media['filename']} ({media_type})")

            enhanced_prompt = f"{prompt}\n\nAvailable uploaded media: {', '.join(uploaded_files)}"
            logger.info(f"Enhanced prompt with uploaded media: {uploaded_files}")
        else:
            logger.info("Uploaded media present but not relevant to current query - not adding to prompt")

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )  # add user message to chat history
    enhanced_prompt = enhanced_prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {enhanced_prompt}")

    with st.chat_message("assistant"):
        sessionState = ""
        chat.references = []
        chat.image_url = []
        response, images = chat.run_multi_agent_system(enhanced_prompt, "Enable", st)

    # Add the assistant message with any images
    assistant_message = {"role": "assistant", "content": response}
    if images:
        logger.info(f"Adding {len(images)} images to assistant message")
        assistant_message["images"] = images
    else:
        logger.info("No images to add to assistant message")

    st.session_state.messages.append(assistant_message)
    logger.info(f"Assistant message added with {len(assistant_message.get('images', []))} images")

    # Debug: Print the full message structure
    if 'images' in assistant_message:
        logger.info("DETAILED MESSAGE DEBUG:")
        logger.info(f"Message role: {assistant_message['role']}")
        logger.info(f"Message content length: {len(assistant_message['content'])}")
        logger.info(f"Number of images: {len(assistant_message['images'])}")
        for img_idx, img in enumerate(assistant_message['images']):
            logger.info(f"  Image {img_idx}: type={type(img)}")
            if isinstance(img, dict):
                logger.info(f"    Keys: {list(img.keys())}")
                if 's3_key' in img:
                    logger.info(f"    S3 key: {img['s3_key']}")
                if 'caption' in img:
                    logger.info(f"    Caption: {img['caption']}")

    # Force a rerun to trigger display_chat_messages
    logger.info("About to trigger display_chat_messages by rerunning...")
    st.rerun()
