"""
AWS CV MCP Server - Multi-Agent Framework Tool Provider

This MCP server exposes computer vision tools to the multi-agent framework.
It serves as the bridge between agents and AWS services (Bedrock, S3, Rekognition).

🏗️ FRAMEWORK INTEGRATION:
- Agents connect to this server via MCP protocol
- Tools defined in cv_tools.py are automatically exposed
- Server handles AWS service configuration and connections
- Tools appear in agent.list_tools() for dynamic usage

🔧 EXTENSION PATTERN:

1. Add New Tools:
   - Create tool function in cv_tools.py with @server.tool() decorator
   - Import the function in this file
   - Tool automatically becomes available to all agents

2. Tool Registration:
   from cv_tools import new_tool_function
   # Function decorator handles registration automatically

3. Configuration:
   - AWS credentials via environment variables or IAM roles
   - S3 bucket and region configuration in connections.py
   - Model settings in bedrock_utils.py

📋 CURRENTLY EXPOSED TOOLS:
- describe_image: Image description using Bedrock Claude
- analyze_video: Video analysis and monitoring
- detect_labels: Object/scene detection using Rekognition
- crop_bounding_box: Extract objects from images
- remove_background: Background removal processing

🔄 DATA FLOW:
Agent Request → MCP Protocol → Server → AWS Services → Response → Agent

This server enables the multi-agent framework's computer vision capabilities.
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Union

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# 🛠️ TOOL IMPORTS - Framework Extension Point
# Add new tools here after creating them in cv_tools.py
from .cv_tools import (
    analyze_video,
    crop_bounding_box,
    describe_image,
    detect_labels,
    remove_background,
    segment_anything,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("aws_cv_mcp_server")

# Create the MCP server
mcp = FastMCP(
    'aws-cv-mcp-server',
    dependencies=[
        'pydantic',
        'boto3',
    ],
    log_level='ERROR',
    instructions="""Use this server to carry out various computer vision tasks.""",
)

@mcp.tool(name='describe_image')
async def mcp_describe_image(
    image_file_name: str = Field(
        ...,
        description='The name of the image file in S3 to analyze. This should be the full path/key in the S3 bucket.',
    ),
    monitoring_instructions: str = Field(
        ...,
        description='Anything specific to look for',
    ),
) -> Dict[str, Union[List[str], str]]:
    """Analyze an image using Amazon Bedrock's Claude model.

    Use this to get an understanding of the content of the image.
    This tool takes an image from S3 and uses Claude to analyze it
    The image is processed through Amazon Bedrock's Claude model to generate a detailed description and analysis.

    USAGE INSTRUCTIONS:
    1. Provide the S3 image file name (key) that you want to analyze
    2. Provide specific monitoring instructions to guide the analysis
    3. The tool will return a detailed analysis of the image based on the instructions

    Returns:
        Dictionary containing the source image file name and the analysis results
    """
    try:
        logger.debug(f"Analyzing image: {image_file_name}")
        # Use the describe_image function from cv_tools
        response = await describe_image(
            image_file_name=image_file_name,
            monitoring_instructions=monitoring_instructions
        )
        return {
            "description": response.analysis,
            "source_image": image_file_name
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool(name='analyze_video')
async def mcp_analyze_video(
    video_file_name: str = Field(
        ...,
        description='The name of the video file in S3 to analyze. This should be the full path/key in the S3 bucket.',
    ),
    monitoring_instructions: str = Field(
        ...,
        description=(
            'Specific instructions for what to monitor or analyze in the video. '
            'This guides the model on what aspects to focus on.'
        ),
    ),
    model_id: str = Field(
        default="us.amazon.nova-lite-v1:0",
        description=(
            'The Amazon Nova model to use for analysis. Options: '
            'us.amazon.nova-lite-v1:0, us.amazon.nova-pro-v1:0, us.amazon.nova-premier-v1:0'
        ),
    ),
) -> Dict[str, Union[str, float, int]]:
    """Analyze a video using Amazon Nova model.

    This tool takes a video from S3 and uses Amazon Nova to analyze it according to specific monitoring instructions.
    The video is processed through Amazon Nova's video understanding capabilities to generate a detailed analysis.

    USAGE INSTRUCTIONS:
    1. Provide the S3 video file name (key) that you want to analyze
    2. Provide specific monitoring instructions to guide the analysis
    3. Optionally specify which Nova model to use (default: nova-lite-v1)
    4. The tool will return a detailed analysis of the video based on the instructions

    Returns:
        Dictionary containing the analysis results, video metrics, and source information
    """
    try:
        # Use the analyze_video function from cv_tools
        response = await analyze_video(
            video_file_name=video_file_name,
            monitoring_instructions=monitoring_instructions,
            model_id=model_id
        )
        return {
            "analysis": response.analysis,
            "source_video": response.source,
            "model_used": response.model_used,
            "video_duration": response.video_duration,
            "frames_analyzed": response.frames_analyzed,
            "sampling_rate": response.sampling_rate,
            "status": response.status,
            "message": response.message
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(name='detect_labels')
async def mcp_detect_labels(
    prefix: str = Field(
        ...,
        description='The name of the image file in S3 to analyze. This should be the full path/key in the S3 bucket.',
    ),
) -> Dict[str, Union[str, List[Dict], float]]:
    """This will return a list of detected object with bounding boxes in an image using Amazon Rekognition.

    Example response item:

    {
      "name": "Child",
      "confidence": 99.22,
      "instances": [
        {
          "bounding_box": {
            "width": 0.17,
            "height": 0.91,
            "left": 0.27,
            "top": 0.05
          },
          "confidence": 99.22,
          "dominant_colors": []
        }
      ],
      "parents": ["Person"],
      "aliases": ["Kid"],
      "categories": ["Person Description"]
    }



    Returns:
        Dictionary containing the detected labels, image properties, and analysis results
    """
    try:
        logger.debug(f"Detecting labels in image: {prefix}")
        response = await detect_labels(
            prefix=prefix,
        )
        return response
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(name='crop_bounding_box')
async def mcp_crop_bounding_box(
#def crop_bounding_box(
    image_file_name: str = Field(..., description="S3 key of the image to crop"),
    bounding_box: Dict[str, float] = Field(
        ...,
        description="Normalized bounding box with keys: left, top, width, height"
    )
) -> Dict[str, str]:
    """
    Crop a specific bounding box from the image and return base64-encoded cropped image.
    """
    try:
        logger.debug(f"Cropping image: {image_file_name} with box: {bounding_box}")
        response = await crop_bounding_box(
            image_file_name=image_file_name,
            bounding_box=bounding_box
        )
        return response

    except Exception as e:
        return {"error": str(e)}


@mcp.tool(name='remove_background')
async def remove_background_tool(
    image_file_name: str = Field(
        ...,
        description="Name of the image file in S3 to remove background from"
    ),
    model_name: str = Field(
        default="u2net",
        description="Background removal model to use (u2net, u2net_human_seg, silueta, etc.)"
    )
) -> Dict[str, str]:
    """
    Remove background from an image using AI-powered background removal.
    Returns the processed image with transparent background.
    """
    try:
        logger.debug(f"Removing background from: {image_file_name} using model: {model_name}")
        response = await remove_background(
            image_file_name=image_file_name,
            model_name=model_name
        )
        return response

    except Exception as e:
        return {"error": str(e)}


@mcp.tool(name='segment_anything')
async def segment_anything_tool(
    image_file_name: str = Field(
        ...,
        description="Name of the image file in S3 to segment using SAM (Segment Anything Model)"
    ),
    model_type: str = Field(
        default="vit_b",
        description="SAM model variant: 'vit_b' (fastest), 'vit_l' (balanced), 'vit_h' (highest quality)"
    ),
    return_masks: bool = Field(
        default=True,
        description="Whether to return colored mask overlay on the segmented image"
    )
) -> Dict[str, Any]:
    """
    Advanced image segmentation using Meta's Segment Anything Model (SAM).

    This tool automatically detects and segments ALL objects in an image, returning:
    - A segmented image with colored masks overlaid on detected objects
    - Bounding box coordinates for each detected segment
    - Quality metrics for each segment (area, stability score)

    SAM is state-of-the-art for automatic image segmentation and can detect objects
    that traditional object detection might miss. Perfect for detailed image analysis,
    object extraction workflows, and understanding image composition.

    Usage Examples:
    - segment_anything("photo.jpg") → Segments all objects with default fast model
    - segment_anything("complex_scene.png", "vit_h") → High-quality segmentation
    - segment_anything("document.jpg", "vit_l") → Balanced speed/quality for documents

    The tool returns the S3 key of the segmented image for display, plus detailed
    bounding box data that can be used with other tools like crop_bounding_box.
    """
    try:
        logger.debug(f"Segmenting image: {image_file_name} with SAM model: {model_type}")
        response = await segment_anything(
            image_file_name=image_file_name,
            model_type=model_type,
            return_masks=return_masks
        )
        return response

    except Exception as e:
        return {"error": str(e)}

def main():
    """Run the MCP server with CLI argument support."""
    parser = argparse.ArgumentParser(
        description='An MCP server that provides computer vision capabilities using Amazon Bedrock'
    )
    parser.add_argument('--sse', action='store_true', help='Use SSE transport')
    parser.add_argument('--port', type=int, default=8888, help='Port to run the server on')

    args = parser.parse_args()

    # Run server with appropriate transport
    if args.sse:
        mcp.settings.port = args.port
        mcp.run(transport='sse')
    else:
        mcp.run()

if __name__ == '__main__':
    main()
