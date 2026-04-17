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

"""Models used by the computer vision MCP server."""

from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel

class ImageAnalysisResponse(BaseModel):
    """Response model for image analysis."""
    status: Literal['success', 'error']
    source: str
    analysis: Optional[str] = None
    message: str

class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis using Amazon Nova."""
    status: Literal['success', 'error']
    source: str
    analysis: Optional[str] = None
    message: str
    model_used: str = "amazon.nova-lite-v1"  # Default to Nova Lite model
    video_duration: Optional[float] = None  # Duration in seconds
    frames_analyzed: Optional[int] = None
    sampling_rate: Optional[float] = None  # Frames per second analyzed

class BoundingBox(BaseModel):
    """Bounding box for detected object."""
    width: float
    height: float
    left: float
    top: float

class LabelInstance(BaseModel):
    """Instance of a detected label with bounding box."""
    bounding_box: BoundingBox
    confidence: float
    dominant_colors: Optional[List[Dict[str, Union[str, float]]]] = None

class LabelCategory(BaseModel):
    """Category information for a label."""
    name: str

class LabelParent(BaseModel):
    """Parent label information."""
    name: str

class LabelAlias(BaseModel):
    """Alias information for a label."""
    name: str

class DetectedLabel(BaseModel):
    """A label detected in an image."""
    name: str
    confidence: float
    instances: List[LabelInstance] = []
    parents: List[LabelParent] = []
    aliases: List[LabelAlias] = []
    categories: List[LabelCategory] = []

class ImageQuality(BaseModel):
    """Image quality metrics."""
    brightness: Optional[float] = None
    sharpness: Optional[float] = None
    contrast: Optional[float] = None

class DominantColor(BaseModel):
    """Dominant color information."""
    red: int
    green: int
    blue: int
    hex_code: str
    simplified_color: str
    css_color: Optional[str] = None
    pixel_percentage: float

class ImageProperties(BaseModel):
    """Image properties including quality and colors."""
    quality: Optional[ImageQuality] = None
    dominant_colors: Optional[List[DominantColor]] = None
    foreground: Optional[Dict[str, Union[ImageQuality, List[DominantColor]]]] = None
    background: Optional[Dict[str, Union[ImageQuality, List[DominantColor]]]] = None

class LabelDetectionResponse(BaseModel):
    """Response model for label detection."""
    status: Literal['success', 'error']
    source: str
    labels: List[DetectedLabel] = []
    image_properties: Optional[ImageProperties] = None
    label_model_version: Optional[str] = None
    message: str
