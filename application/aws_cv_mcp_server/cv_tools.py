#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the 'license' file accompanying this file. This file is distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions
# and limitations under the License.
#

"""
Computer Vision Tools for Multi-Agent Framework

This module provides the core CV tool implementations that power the multi-agent system.
These tools are exposed via MCP (Model Context Protocol) and used by CV agents.

🏗️ FRAMEWORK TOOL ARCHITECTURE:
- Each function decorated with @server.tool() becomes available to agents
- Tools handle AWS service interactions (Bedrock, S3, Rekognition)
- Direct memory processing eliminates temporary files
- S3-based image display for streamlit UI integration

🔧 EXTENSION PATTERNS:

1. Adding New CV Tools:
   @server.tool()
   async def new_cv_tool(parameter: str) -> str:
       '''Clear description for agents to understand usage'''
       # Implementation using AWS services
       return result

2. Tool Design Principles:
   - Clear docstrings for agent understanding
   - Consistent return formats (success messages or error descriptions)
   - S3 integration for image persistence and display
   - Robust error handling with descriptive messages

3. Agent Integration:
   - Tools automatically appear in agent.list_tools()
   - Agent prompts reference tools by name and description
   - Tools handle technical details, agents handle coordination

📋 AVAILABLE TOOL CATEGORIES:
- Image Analysis: describe_image, detect_labels
- Image Processing: crop_bounding_box, remove_background
- Video Analysis: analyze_video
- UI Integration: ui_show_image, ui_show_images

🔄 DATA FLOW:
Input (Agent) → Tool Function → AWS Services → S3 Storage → UI Display → Output (Agent)

🧪 TESTING:
- Each tool has unit tests in tests/test_cv_tools.py
- Tests mock AWS services for reliable testing
- Follow naming: test_[tool_name]_[scenario]
"""

import logging
import os
import json
import re
import uuid
import sys
import time
import urllib.request
import colorsys
from typing import Literal, Dict, Any
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
from rembg import remove, new_session
import boto3
from segment_anything import sam_model_registry
from application.aws_cv_mcp_server.connections import Connections
from application.aws_cv_mcp_server.models import ImageAnalysisResponse, \
    VideoAnalysisResponse, LabelDetectionResponse, DetectedLabel, LabelInstance, \
    BoundingBox, ImageProperties
from application.aws_cv_mcp_server.bedrock_utils import create_multimodal_prompt, invoke_bedrock_model


logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("cv_tools")


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


# SEC: H4 — Rate limiters for expensive AI operations
_video_limiter = RateLimiter(max_calls=5, period_seconds=60)
_sam_limiter = RateLimiter(max_calls=3, period_seconds=60)


def _sanitize_s3_key(key: str) -> str:
    """Sanitize S3 key to prevent path traversal (CWE-22).

    Strips traversal sequences and ensures the key stays within the mcp/ prefix.
    """
    # Remove path traversal attempts
    key = key.replace('../', '').replace('..\\', '')
    # Remove null bytes
    key = key.replace('\x00', '')
    # Strip leading slashes
    key = key.lstrip('/')
    # Use only the basename to prevent directory escape
    key = os.path.basename(key)
    return key


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe S3 storage (CWE-22).

    Removes dangerous characters and limits length.
    """
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename[:255]


def _is_url(file_name: str) -> bool:
    """Check if the file name is actually a URL"""
    return (file_name.startswith("http://") or
            file_name.startswith("https://") or
            ("." in file_name and "/" in file_name and not file_name.startswith("mcp/")))


async def describe_image(image_file_name: str, monitoring_instructions: str) -> ImageAnalysisResponse:
    """Analyze an image using Amazon Bedrock's Claude model.

    Args:
        image_file_name: The name of the image file in S3 to analyze
        monitoring_instructions: Specific instructions for what to monitor or analyze in the image

    Returns:
        ImageAnalysisResponse: Response containing the analysis results
    """
    # Validate input - reject URLs immediately
    if _is_url(image_file_name):
        return ImageAnalysisResponse(
            status="error",
            source=image_file_name,
            analysis=None,
            message=(
                f"ERROR: CV tools cannot process URLs. "
                f"Use image_opensearch_agent for URL-based images: {image_file_name}"
            )
        )

    try:
        # Get the image from S3

        prefix = f"mcp/{image_file_name}"
        logger.debug(f"Attempting to analyze image at s3://{Connections.agent_bucket_name}/{prefix}")
        response = Connections.s3_client.get_object(
            Bucket=Connections.agent_bucket_name,
            Key=prefix
        )
        image_data = response['Body'].read()
        content_type = response['ContentType']

        # Ensure content type is supported by Bedrock
        valid_content_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        if content_type not in valid_content_types:
            # Try to determine from file extension
            if image_file_name.lower().endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif image_file_name.lower().endswith('.png'):
                content_type = 'image/png'
            elif image_file_name.lower().endswith('.gif'):
                content_type = 'image/gif'
            elif image_file_name.lower().endswith('.webp'):
                content_type = 'image/webp'
            else:
                content_type = 'image/jpeg'  # Default fallback
            logger.debug(f"Content type corrected to: {content_type}")

        # Create the prompt for Claude
        prompt = create_multimodal_prompt(
            image_data=image_data,
            text=monitoring_instructions,
            content_type=content_type,
            system_prompt=(
                "You are a helpful assistant that analyzes images. "
                "Provide detailed, accurate descriptions based on the user's instructions."
            )
        )

        # Get analysis from Claude
        analysis = invoke_bedrock_model(prompt)

        return ImageAnalysisResponse(
            status="success",
            source=prefix,
            analysis=analysis,
            message="Image analysis completed successfully"
        )

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return ImageAnalysisResponse(
            status="error",
            source=image_file_name,
            analysis=None,
            message="Image analysis failed. Please check the image and try again."
        )

async def analyze_video(
    video_file_name: str,
    monitoring_instructions: str,
    model_id: Literal[
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-premier-v1:0"
    ] = "us.amazon.nova-lite-v1:0",
) -> VideoAnalysisResponse:
    """Analyze a video using Amazon Nova model.

    Args:
        video_file_name: The name of the video file in S3 to analyze
        monitoring_instructions: Specific instructions for what to monitor or analyze in the video
        model_id: The Nova model to use for analysis (default: us.amazon.nova-lite-v1:0)

    Returns:
        VideoAnalysisResponse: Response containing the analysis results
    """
    logger.debug("=== ANALYZE_VIDEO FUNCTION CALLED ===")
    logger.debug(f"Function parameters: video_file_name={video_file_name}, model_id={model_id}")

    # SEC: H4 — Rate limit expensive video analysis
    if not _video_limiter.allow():
        return VideoAnalysisResponse(
            status="error",
            source=video_file_name,
            analysis=None,
            message="Rate limit exceeded. Please wait before submitting another video.",
            model_used=model_id
        )

    try:
        logger.debug(f"Video file: {video_file_name}")
        logger.debug(f"Connections.region_name: {Connections.region_name}")
        logger.debug(f"Connections.agent_bucket_name: {Connections.agent_bucket_name}")
        logger.debug(f"S3 key will be: mcp/{video_file_name}")
        logger.debug(f"Full S3 URI: s3://{Connections.agent_bucket_name}/mcp/{video_file_name}")

        # Use the properly configured Bedrock client from Connections class
        bedrock_client = Connections.bedrock_client

        # Construct the S3 key with proper prefix
        s3_key = f"mcp/{video_file_name}"

        # Validate video file exists in S3 before calling Bedrock
        try:
            logger.debug("Checking if video exists in S3...")
            Connections.s3_client.head_object(
                Bucket=Connections.agent_bucket_name,
                Key=s3_key
            )
            logger.debug("Video file confirmed to exist in S3")
        except Exception as s3_error:
            # SEC: H7 — Log detailed error internally, return generic message (CWE-209)
            logger.error(
                f"Video file not found in S3: "
                f"s3://{Connections.agent_bucket_name}/{s3_key}. "
                f"Error: {str(s3_error)}"
            )
            return VideoAnalysisResponse(
                status="error",
                source=video_file_name,
                analysis=None,
                message="Video file not found. Please check the filename and try again.",
                model_used=model_id
            )

        # Validate video format against allowlist
        ALLOWED_VIDEO_FORMATS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        file_extension = video_file_name.rsplit('.', 1)[-1].lower()
        if file_extension not in ALLOWED_VIDEO_FORMATS:
            error_message = f"Unsupported video format: {file_extension}. Supported formats: {', '.join(sorted(ALLOWED_VIDEO_FORMATS))}"
            logger.error(error_message)
            return VideoAnalysisResponse(
                status="error",
                source=video_file_name,
                analysis=None,
                message=error_message,
                model_used=model_id
            )

        logger.debug(f"Video format '{file_extension}' is supported")

        # Prepare the system prompt
        system_list = [{
            "text": (
                "You are an expert video analyst. "
                "Analyze the video according to the user's instructions "
                "and provide detailed insights."
            )
        }]

        # Prepare the message with S3 video source
        message_list = [{
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": file_extension,
                        "source": {
                            "s3Location": {
                                "uri": f"s3://{Connections.agent_bucket_name}/{s3_key}"
                            }
                        }
                    }
                },
                {
                    "text": monitoring_instructions
                }
            ]
        }]

        # Configure inference parameters
        inference_config = {
            "maxTokens": 1000,
            "temperature": 0.3,
            "topP": 0.9
        }

        # Prepare the request
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inference_config
        }

        # Get video metadata from S3
        s3_response = Connections.s3_client.head_object(
            Bucket=Connections.agent_bucket_name,
            Key=s3_key
        )

        logger.debug(f"Calling Bedrock Nova model: {model_id}")
        logger.debug(f"Request body size: {len(json.dumps(request_body))} bytes")

        # Invoke Nova model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )

        logger.debug("Bedrock Nova model responded successfully")

        model_response = json.loads(response["body"].read())
        analysis = model_response["output"]["message"]["content"][0]["text"]

        # Calculate video metrics based on Nova's sampling strategy
        video_duration = s3_response.get('Metadata', {}).get('duration', 0)
        frames_analyzed = None
        sampling_rate = None

        if video_duration:
            video_duration = float(video_duration)
            if video_duration <= 960:  # Less than 16 minutes
                frames_analyzed = int(video_duration)
                sampling_rate = 1.0
            else:
                frames_analyzed = 960
                sampling_rate = 960 / video_duration

        return VideoAnalysisResponse(
            status="success",
            source=video_file_name,
            analysis=analysis,
            message="Video analysis completed successfully",
            model_used=model_id,
            video_duration=video_duration,
            frames_analyzed=frames_analyzed if video_duration else None,
            sampling_rate=sampling_rate if video_duration else None
        )

    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return VideoAnalysisResponse(
            status="error",
            source=video_file_name,
            analysis=None,
            message="Video analysis failed. Please check the file and try again.",
            model_used=model_id
        )

async def detect_labels(prefix: str) -> LabelDetectionResponse:  # pylint: disable=too-many-branches,too-many-statements
    """Detect labels with bounding boxes in an image using Amazon Rekognition.

    Args:
        prefix: The name of the image file in S3 to analyze

    Returns:
        LabelDetectionResponse: Response containing detected labels and image properties
    """
    # Validate input - reject URLs immediately
    if _is_url(prefix):
        return LabelDetectionResponse(
            status="error",
            source=prefix,
            message=f"ERROR: CV tools cannot process URLs. Use image_opensearch_agent for URL-based images: {prefix}"
        )

    try:
        logger.debug(f"detect_labels: bucket='{Connections.agent_bucket_name}', "
                     f"file='{prefix}', path=s3://{Connections.agent_bucket_name}/{prefix}")
        settings = None
        prefix = "mcp/" + prefix if not prefix.startswith("mcp/") else prefix
        try:
            # Set default settings
            if settings is None:
                settings = {}

            max_labels = settings.get("max_labels", 10)
            min_confidence = settings.get("min_confidence", 75.0)
            include_image_properties = settings.get("include_image_properties", True)
            label_filters = settings.get("label_filters", {})
            label_inclusion_filters = label_filters.get("include")
            label_exclusion_filters = label_filters.get("exclude")

            # Create Rekognition client
            rekognition_client = boto3.client(
                'rekognition',
                region_name=Connections.region_name
            )

            # Log the S3 path we're trying to access
            logger.debug(f"Attempting to access S3 object - Bucket: {Connections.agent_bucket_name}, Key: {prefix}")

            # Prepare request parameters
            request_params = {
                'Image': {
                    'S3Object': {
                        'Bucket': Connections.agent_bucket_name,
                        'Name': prefix
                    }
                },
                'MaxLabels': max_labels,
                'MinConfidence': min_confidence,
            }

            # Add features and settings if needed
            if include_image_properties or label_inclusion_filters or label_exclusion_filters:
                request_params['Features'] = ['GENERAL_LABELS']
                if include_image_properties:
                    request_params['Features'].append('IMAGE_PROPERTIES')

                settings = {}
                if label_inclusion_filters or label_exclusion_filters:
                    settings['GeneralLabels'] = {}
                    if label_inclusion_filters:
                        settings['GeneralLabels']['LabelInclusionFilters'] = label_inclusion_filters
                    if label_exclusion_filters:
                        settings['GeneralLabels']['LabelExclusionFilters'] = label_exclusion_filters

                if include_image_properties:
                    settings['ImageProperties'] = {'MaxDominantColors': 10}

                if settings:
                    request_params['Settings'] = settings

            # Call Rekognition API
            response = rekognition_client.detect_labels(**request_params)

            # Debug log the response structure
            logger.debug(f"Rekognition response structure: {json.dumps(response, indent=2)}")

            # Process response
            labels = []
            for label in response.get('Labels', []):
                instances = []
                for instance in label.get('Instances', []):
                    # Convert capitalized bounding box fields to lowercase
                    bbox = instance['BoundingBox']
                    bbox_lower = {
                        'width': bbox['Width'],
                        'height': bbox['Height'],
                        'left': bbox['Left'],
                        'top': bbox['Top']
                    }
                    instances.append(LabelInstance(
                        bounding_box=BoundingBox(**bbox_lower),
                        confidence=instance['Confidence'],
                        dominant_colors=instance.get('DominantColors')
                    ))

                labels.append(DetectedLabel(
                    name=label['Name'],
                    confidence=label['Confidence'],
                    instances=instances,
                    parents=[{'name': p['Name']} for p in label.get('Parents', [])],
                    aliases=[{'name': a['Name']} for a in label.get('Aliases', [])],
                    categories=[{'name': c['Name']} for c in label.get('Categories', [])]
                ))

            # Process image properties if included
            image_properties = None
            if 'ImageProperties' in response:
                props = response['ImageProperties']
                logger.debug(f"Image properties structure: {json.dumps(props, indent=2)}")

                def convert_color(color_dict):
                    """Convert Rekognition color format to our model format."""
                    logger.debug(f"Color dict structure: {json.dumps(color_dict, indent=2)}")
                    return {
                        'red': color_dict['Red'],
                        'green': color_dict['Green'],
                        'blue': color_dict['Blue'],
                        'hex_code': f"#{color_dict['Red']:02x}{color_dict['Green']:02x}{color_dict['Blue']:02x}",
                        'simplified_color': f"rgb({color_dict['Red']}, {color_dict['Green']}, {color_dict['Blue']})",
                        # Try to get Percentage, default to 0.0 if not found
                        'pixel_percentage': color_dict.get('Percentage', 0.0)
                    }

                def process_color_list(colors):
                    """Process a list of colors from Rekognition."""
                    if not colors:
                        return None
                    return [convert_color(color) for color in colors]

                # Process dominant colors
                dominant_colors = process_color_list(props.get('DominantColors', []))

                # Process foreground and background if present
                foreground = None
                background = None

                if 'Foreground' in props:
                    fg = props['Foreground']
                    foreground = {
                        'quality': fg.get('Quality'),
                        'dominant_colors': process_color_list(fg.get('DominantColors', []))
                    }

                if 'Background' in props:
                    bg = props['Background']
                    background = {
                        'quality': bg.get('Quality'),
                        'dominant_colors': process_color_list(bg.get('DominantColors', []))
                    }

                image_properties = ImageProperties(
                    quality=props.get('Quality'),
                    dominant_colors=dominant_colors,
                    foreground=foreground,
                    background=background
                )
                # list labels to temp file
                # with open("labels.json", "w") as f:
                #     f.write(str(labels))

            return LabelDetectionResponse(
                status="success",
                source=prefix,
                labels=labels,
                image_properties=image_properties,
                label_model_version=response.get('LabelModelVersion'),
                message="Label detection completed successfully"
            )

        except Exception as e:
            s3_path = f"s3://{Connections.agent_bucket_name}/{prefix}"
            # SEC: H7 — Log detailed error internally, return generic message (CWE-209)
            logger.error(f"Error detecting labels for {s3_path}: {str(e)}")
            return LabelDetectionResponse(
                status="error",
                source=prefix,
                message="Label detection failed. Please check the image and try again."
            )
    except Exception as e:
        # SEC: Log detailed error internally, return generic message to user (CWE-209)
        logger.error(f"Unexpected error in detect_labels: {str(e)}")
        return LabelDetectionResponse(
            status="error",
            source=prefix,
            message="An unexpected error occurred during label detection. Please try again."
        )

async def crop_bounding_box(image_file_name: str, bounding_box: Dict[str, float]) -> Dict[str, str]:
    """
    Crop a specific bounding box from the image and return base64-encoded cropped image.
    """
    logger.debug(f"crop_bounding_box called: file={image_file_name}, bbox={bounding_box}")

    # Generate unique identifier for this crop operation
    crop_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())

    prefix = "mcp/" + image_file_name
    try:
        s3 = Connections.s3_client
        bucket = Connections.agent_bucket_name

        obj = s3.get_object(Bucket=bucket, Key=prefix)
        image_bytes = obj['Body'].read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        box = bounding_box
        left = int(box["left"] * width)
        top = int(box["top"] * height)
        right = int((box["left"] + box["width"]) * width)
        bottom = int((box["top"] + box["height"]) * height)

        cropped = image.crop((left, top, right, bottom))

        # Save cropped image to memory buffer
        output = BytesIO()
        cropped.save(output, format="JPEG")

        # Generate unique filename to prevent overwrites
        base_filename = os.path.splitext(os.path.basename(image_file_name))[0]
        extension = os.path.splitext(os.path.basename(image_file_name))[1] or '.png'
        # SEC: Sanitize filename components to prevent path traversal (CWE-22)
        cropped_file_name = _sanitize_filename(f"cropped_{base_filename}_{crop_id}_{timestamp}{extension}")
        cropped_s3_key = f"mcp/{cropped_file_name}"  # Include the mcp/ prefix

        logger.debug(f"Saving cropped image as: {cropped_s3_key}")

        s3.put_object(
            Bucket=bucket,
            Key=cropped_s3_key,
            Body=output.getvalue(),
            ContentType='image/jpeg'
        )

        response_data = {
            "source_image": image_file_name,
            "cropped_bbox": box,
            "cropped_image_filename": cropped_file_name,  # Return just the filename, UI will add mcp/ prefix
            "message": f"Image cropped and saved as {cropped_file_name}.",
            "crop_id": crop_id,
            "unique_filename": cropped_file_name
        }

        logger.debug(f"Returning crop response: {cropped_file_name}")
        return response_data

    except Exception as e:
        # SEC: Log detailed error internally, return generic message to user (CWE-209)
        logger.error(f"Error cropping image: {str(e)}")
        return {"error": "An error occurred while cropping the image. Please try again."}


async def remove_background(image_file_name: str, model_name: str = "u2net") -> Dict[str, Any]:
    """Remove background from an image using AI-powered background removal.

    Args:
        image_file_name: The name of the image file in S3 to process
        model_name: The model to use for background removal (u2net, u2net_human_seg, silueta, etc.)

    Returns:
        Dict containing the processed image data and metadata
    """
    try:
        logger.info(f"Starting background removal for image: {image_file_name}")

        # Construct the S3 key with proper prefix
        s3_key = f"mcp/{image_file_name}" if not image_file_name.startswith("mcp/") else image_file_name

        # Download the image from S3
        response = Connections.s3_client.get_object(
            Bucket=Connections.agent_bucket_name,
            Key=s3_key
        )
        image_data = response['Body'].read()

        # Remove background using rembg
        logger.info(f"Removing background using model: {model_name}")
        # Use rembg's new_session approach for better model control
        session = new_session(model_name)
        output_data = remove(image_data, session=session)

        # Generate unique filename for the processed image
        timestamp = int(time.time())
        base_name = image_file_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace("mcp/", "")
        processed_file_name = f"{base_name}_bg_removed_{timestamp}.png"

        # Upload processed image to S3
        processed_s3_key = f"mcp/{processed_file_name}"
        Connections.s3_client.put_object(
            Bucket=Connections.agent_bucket_name,
            Key=processed_s3_key,
            Body=output_data,
            ContentType='image/png'
        )

        logger.info(f"Background removed image saved to S3: {processed_s3_key}")

        response_data = {
            "source_image": image_file_name,
            "processed_image_filename": processed_file_name,
            "model_used": model_name,
            "message": f"Background removed and saved as {processed_file_name}.",
            "unique_filename": processed_file_name
        }

        return response_data

    except Exception as e:
        logger.error(f"Error in background removal: {str(e)}")
        # SEC: Log detailed error internally, return generic message to user (CWE-209)
        return {"error": "Background removal failed. Please check the image and try again."}


async def segment_anything(  # pylint: disable=too-many-statements,unused-argument
    image_file_name: str,
    model_type: str = "vit_b",
    return_masks: bool = True
) -> Dict[str, Any]:
    """Advanced image segmentation using Segment Anything Model (SAM).

    This tool downloads and uses the SAM model to perform automatic segmentation
    on the entire image, identifying all objects and returning both the segmented
    image and bounding box data for each detected segment.

    Args:
        image_file_name: The name of the image file in S3 to segment
        model_type: SAM model variant to use ("vit_b", "vit_l", "vit_h") - vit_b is fastest
        return_masks: Whether to save and return the segmented masks overlay

    Returns:
        Dict containing:
        - segmented_image_s3_key: S3 key of the segmented image with colored masks
        - original_image_s3_key: S3 key of the original image for reference
        - bounding_boxes: List of bounding box coordinates for each segment
        - segment_count: Number of segments detected
        - model_used: Which SAM model was used
        - message: Success/error message

    Examples:
        segment_anything("photo.jpg") → Returns segmented image with all objects outlined
        segment_anything("document.png", "vit_h") → Uses highest quality model for precise segmentation
    """
    try:
        logger.debug(f"segment_anything called: file={image_file_name}, model={model_type}")

        # SEC: H4 — Rate limit expensive SAM segmentation
        if not _sam_limiter.allow():
            return {
                "status": "error",
                "message": "Rate limit exceeded. Please wait before submitting another segmentation request.",
            }

        # Download SAM model if not already cached
        sam = _get_sam_model(model_type)

        # Generate unique identifiers
        segment_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())

        # Get image from S3
        prefix = f"mcp/{image_file_name}"
        logger.debug(f"Downloading image from S3: s3://{Connections.agent_bucket_name}/{prefix}")

        response = Connections.s3_client.get_object(
            Bucket=Connections.agent_bucket_name,
            Key=prefix
        )
        image_data = response['Body'].read()

        # Convert to OpenCV format
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.debug(f"Image loaded: {image_rgb.shape}")

        # Generate automatic segmentation masks
        from segment_anything import SamAutomaticMaskGenerator  # pylint: disable=import-outside-toplevel
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,  # Dense grid for comprehensive segmentation
            pred_iou_thresh=0.7,  # High quality masks only
            stability_score_thresh=0.8,  # Stable masks only
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Filter out tiny segments
        )

        logger.debug("Generating automatic masks...")
        masks = mask_generator.generate(image_rgb)

        logger.debug(f"Generated {len(masks)} segments")

        # Create visualization with colored masks
        segmented_image = image_rgb.copy()
        bounding_boxes = []

        # Sort masks by area (largest first) for better visualization
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Create colorful overlay
        overlay = np.zeros_like(image_rgb)
        alpha = 0.6  # Increased transparency for more visible overlay (was 0.35)

        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # [x, y, width, height]
            area = mask_data['area']
            stability_score = mask_data['stability_score']

            # Convert to normalized bounding box format (matching existing tools)
            height, width = image_rgb.shape[:2]
            normalized_bbox = {
                "left": bbox[0] / width,
                "top": bbox[1] / height,
                "width": bbox[2] / width,
                "height": bbox[3] / height,
                "area": area,
                "stability_score": float(stability_score)
            }
            bounding_boxes.append(normalized_bbox)

            # Generate a unique, vibrant color for this segment
            color = _generate_segment_color(i, len(masks))

            # Apply colored mask with higher intensity
            colored_mask = np.zeros_like(image_rgb)
            colored_mask[mask] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

            # Draw thicker bounding box outline for better visibility
            x, y, w, h = bbox
            # Increased thickness from 2 to 4
            cv2.rectangle(
                segmented_image,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                color,
                4
            )

        # Combine original image with colored overlay - make overlay more prominent
        # Changed from 0.7/0.3 to 0.5/0.5 for more visible overlay
        final_image = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)

        # Add white outlines around each segment for even better visibility
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']

            # Create white border around each mask
            kernel = np.ones((3,3), np.uint8)
            mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            mask_border = mask_dilated - mask.astype(np.uint8)

            # Add white borders to make segments more visible
            final_image[mask_border > 0] = [255, 255, 255]  # White borders

            # Add colored number labels for each segment (for first 20 segments to avoid clutter)
            if i < 20:
                x, y, w, h = bbox
                center_x, center_y = int(x + w/2), int(y + h/2)
                cv2.putText(final_image, str(i+1), (center_x-10, center_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text with black outline
                cv2.putText(final_image, str(i+1), (center_x-10, center_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)    # Black outline

        # Convert back to PIL for saving
        final_pil = Image.fromarray(final_image)

        # Save segmented image to S3
        segmented_filename = f"segmented_{segment_id}_{timestamp}_{image_file_name}"
        segmented_s3_key = f"mcp/{segmented_filename}"

        # Convert to bytes for S3 upload
        img_buffer = BytesIO()
        final_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Upload segmented image to S3
        Connections.s3_client.put_object(
            Bucket=Connections.agent_bucket_name,
            Key=segmented_s3_key,
            Body=img_buffer.getvalue(),
            ContentType='image/png'
        )

        # Generate S3 URLs
        segmented_s3_url = f"s3://{Connections.agent_bucket_name}/{segmented_s3_key}"
        original_s3_url = f"s3://{Connections.agent_bucket_name}/{prefix}"

        logger.debug(f"Segmented image saved to: {segmented_s3_url}")
        logger.debug(f"Detected {len(bounding_boxes)} segments")

        success_message = (
            f"Successfully segmented image using SAM {model_type.upper()} model. "
            f"Detected {len(bounding_boxes)} distinct segments. "
            f"Segmented image saved as '{segmented_filename}'. "
            f"Each segment has bounding box coordinates and quality metrics."
        )

        return {
            "segmented_image_s3_key": segmented_filename,  # Just filename for UI tools
            "segmented_image_s3_url": segmented_s3_url,
            "original_image_s3_key": image_file_name,
            "original_image_s3_url": original_s3_url,
            "bounding_boxes": bounding_boxes,
            "segment_count": len(bounding_boxes),
            "model_used": f"SAM-{model_type.upper()}",
            "message": success_message,
            "status": "success"
        }

    except Exception as e:
        logger.error(
            f"Error in SAM segmentation: {str(e)}",
            exc_info=True
        )
        return {
            "status": "error",
            # SEC: Log detailed error internally, return generic message to user (CWE-209)
            "message": "SAM segmentation failed. Please check the image and try again.",
        }


def _get_sam_model(model_type: str):
    """Download and cache SAM model. Returns the model for inference."""
    try:
        # Model download URLs (these will be cached locally)
        model_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }

        # SEC: L2 — Known-good SHA256 hashes for integrity verification
        expected_hashes = {
            "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
            "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49f56c",
            "vit_h": "a7bf3b02f3f65b8d44fdc529c8c8a59c80bca04e0e3965f11f1e40eb7c4866f6",
        }

        if model_type not in model_urls:
            model_type = "vit_b"  # Default fallback

        logger.info(f"Loading SAM model: {model_type}")

        # Create cache directory
        cache_dir = "/tmp/sam_models"
        os.makedirs(cache_dir, exist_ok=True)

        model_path = f"{cache_dir}/sam_{model_type}.pth"

        # Download model if not cached
        if not os.path.exists(model_path):
            logger.info(f"Downloading SAM model {model_type}...")
            urllib.request.urlretrieve(model_urls[model_type], model_path)

            # Verify hash after download
            import hashlib  # pylint: disable=import-outside-toplevel
            sha256 = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            digest = sha256.hexdigest()

            if model_type in expected_hashes:
                if digest != expected_hashes[model_type]:
                    os.remove(model_path)
                    raise ValueError(
                        f"SAM model hash mismatch for {model_type}"
                    )
                logger.info("Model hash verified successfully")
            else:
                logger.warning(
                    f"No expected hash for {model_type}, "
                    f"skipping verification"
                )

            logger.info(f"Model downloaded to {model_path}")
        else:
            logger.debug(f"Using cached model at {model_path}")

        # Load model
        sam = sam_model_registry[model_type](checkpoint=model_path)

        # Use CPU if CUDA not available (common in serverless environments)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)

        logger.info(f"SAM model loaded successfully on {device}")
        return sam

    except Exception as e:
        logger.error(f"Error loading SAM model: {e}")
        raise e


def _generate_segment_color(index: int, total_segments: int) -> tuple:
    """Generate a unique, vibrant color for each segment."""

    # Use HSV color space for evenly distributed colors
    hue = (index / max(total_segments, 1)) % 1.0
    saturation = 0.9 + (index % 2) * 0.1  # Higher saturation for more vibrant colors (was 0.7)
    value = 0.9 + (index % 2) * 0.1      # Higher brightness for more visible colors (was 0.8)

    # Convert to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(c * 255) for c in rgb)


if __name__ == "__main__":
    test_image_file = "mcp/image33.png"
    # response = describe_image(
    #     test_image_file,
    #     monitoring_instructions="Describe the image in detail, including objects, actions, and context."
    # )
    # detect_labels(
    #     prefix=test_image_file,
    # )
    crop_bounding_box(test_image_file,
                      {
    "left": 0.41439089179039,
    "top": 0.14356614649295807,
    "width": 0.341442734003067,
    "height": 0.8175641298294067
})
