"""
Test suite for cv_tools module.

This test suite verifies that CV tools:
1. Properly processes images/videos from S3
2. Returns structured results with expected fields
3. Handles error scenarios appropriately
4. Generates unique filenames and proper S3 keys
"""

import asyncio
import os
import sys
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

# Add application directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application", "aws_cv_mcp_server"))

import cv_tools  # pylint: disable=wrong-import-position


class TestCVTools(unittest.TestCase):
    """Test suite for computer vision tools"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_name = "test_image.jpg"
        self.test_video_name = "test_video.mp4"
        self.test_s3_key = f"mcp/{self.test_image_name}"

        # Create mock image data
        self.mock_image = Image.new('RGB', (100, 100), color='red')
        self.mock_image_bytes = BytesIO()
        self.mock_image.save(self.mock_image_bytes, format='JPEG')
        self.mock_image_data = self.mock_image_bytes.getvalue()

        # Mock S3 response
        self.mock_s3_response = {
            'Body': Mock(),
            'ContentType': 'image/jpeg'
        }
        self.mock_s3_response['Body'].read.return_value = self.mock_image_data

    @patch('cv_tools.Connections')
    def test_describe_image_success(self, mock_connections):
        """Test successful image description"""
        async def async_test():
            # Setup mocks
            mock_connections.s3_client.get_object.return_value = self.mock_s3_response
            mock_connections.agent_bucket_name = "test-bucket"

            with patch('cv_tools.invoke_bedrock_model') as mock_bedrock:
                mock_bedrock.return_value = "A red square image"

                result = await cv_tools.describe_image(
                    self.test_image_name,
                    "Describe this image"
                )

                # Verify S3 was called correctly
                mock_connections.s3_client.get_object.assert_called_once_with(
                    Bucket="test-bucket",
                    Key=self.test_s3_key
                )

                # Verify result structure
                self.assertEqual(result.status, "success")
                self.assertEqual(result.source, f"mcp/{self.test_image_name}")
                self.assertEqual(result.analysis, "A red square image")

        # Run the async test
        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    def test_crop_bounding_box_success(self, mock_connections):
        """Test successful image cropping"""
        async def async_test():
            # Setup mocks
            mock_connections.s3_client.get_object.return_value = self.mock_s3_response
            mock_connections.s3_client.put_object.return_value = {}
            mock_connections.agent_bucket_name = "test-bucket"

            bounding_box = {
                "left": 0.1,
                "top": 0.1,
                "width": 0.5,
                "height": 0.5
            }

            result = await cv_tools.crop_bounding_box(
                self.test_image_name,
                bounding_box
            )

            # Verify result structure
            self.assertIn("source_image", result)
            self.assertIn("cropped_image_filename", result)
            self.assertIn("unique_filename", result)
            self.assertEqual(result["source_image"], self.test_image_name)

        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    @patch('cv_tools.remove')
    @patch('cv_tools.new_session')
    def test_remove_background_success(self, mock_new_session, mock_remove, mock_connections):
        """Test successful background removal"""
        async def async_test():
            # Setup mocks
            mock_connections.s3_client.get_object.return_value = self.mock_s3_response
            mock_connections.s3_client.put_object.return_value = {}
            mock_connections.agent_bucket_name = "test-bucket"

            mock_session = Mock()
            mock_new_session.return_value = mock_session
            mock_remove.return_value = self.mock_image_data

            result = await cv_tools.remove_background(
                self.test_image_name,
                "u2net"
            )

            # Verify rembg was called correctly
            mock_new_session.assert_called_once_with("u2net")
            mock_remove.assert_called_once_with(self.mock_image_data, session=mock_session)

            # Verify result structure
            self.assertIn("source_image", result)
            self.assertIn("processed_image_filename", result)
            self.assertIn("model_used", result)
            self.assertEqual(result["source_image"], self.test_image_name)
            self.assertEqual(result["model_used"], "u2net")

        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    def test_detect_labels_success(self, mock_connections):
        """Test successful label detection"""
        async def async_test():
            # Setup mocks
            mock_connections.s3_client.head_object.return_value = {}
            mock_connections.agent_bucket_name = "test-bucket"

            mock_rekognition = Mock()
            mock_rekognition.detect_labels.return_value = {
                'Labels': [
                    {
                        'Name': 'Dog',
                        'Confidence': 95.5,
                        'Instances': [
                            {
                                'BoundingBox': {
                                    'Left': 0.1,
                                    'Top': 0.1,
                                    'Width': 0.5,
                                    'Height': 0.5
                                },
                                'Confidence': 95.5
                            }
                        ]
                    }
                ]
            }

            with patch('boto3.client', return_value=mock_rekognition):
                result = await cv_tools.detect_labels(self.test_s3_key)

                # Verify result structure
                self.assertEqual(result.status, "success")
                self.assertEqual(len(result.labels), 1)
                self.assertEqual(result.labels[0].name, "Dog")
                self.assertEqual(result.labels[0].confidence, 95.5)

        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    def test_analyze_video_success(self, mock_connections):
        """Test successful video analysis"""
        async def async_test():
            # Setup mocks
            mock_connections.s3_client.head_object.return_value = {
                'Metadata': {'duration': '30.0'}
            }
            mock_connections.agent_bucket_name = "test-bucket"
            mock_connections.region_name = "us-east-1"

            mock_bedrock = Mock()
            mock_bedrock.invoke_model.return_value = {
                'body': Mock()
            }
            mock_response_body = {
                'output': {
                    'message': {
                        'content': [
                            {'text': 'Video shows a person walking with a dog'}
                        ]
                    }
                }
            }
            mock_bedrock.invoke_model.return_value['body'].read.return_value = Mock()

            with patch('boto3.client', return_value=mock_bedrock), \
                 patch('json.loads', return_value=mock_response_body):

                result = await cv_tools.analyze_video(
                    self.test_video_name,
                    "Analyze this video for activities"
                )

                # Verify result structure
                self.assertEqual(result.status, "success")
                self.assertEqual(result.source, self.test_video_name)
                self.assertIn("person walking", result.analysis)

        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    def test_error_handling_invalid_s3_key(self, mock_connections):
        """Test error handling for invalid S3 key"""
        async def async_test():
            # Setup mocks
            mock_connections.s3_client.get_object.side_effect = Exception("S3 object not found")

            result = await cv_tools.describe_image(
                "nonexistent.jpg",
                "Describe this image"
            )

            # Verify error handling — generic message returned (H7)
            self.assertEqual(result.status, "error")
            self.assertIn("Image analysis failed", result.message)

        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    def test_segment_anything_success(self, mock_connections):
        """Test successful SAM segmentation"""
        async def async_test():
            # Setup mocks
            mock_connections.agent_bucket_name = "test-bucket"
            mock_connections.s3_client.get_object.return_value = self.mock_s3_response
            mock_connections.s3_client.put_object.return_value = {"ETag": "test-etag"}

            # Mock the entire segment_anything import and related functions
            with patch('cv_tools._get_sam_model') as mock_get_sam_model, \
                 patch('cv_tools.np.frombuffer') as mock_frombuffer, \
                 patch('cv_tools.cv2.imdecode') as mock_imdecode, \
                 patch('cv_tools.cv2.cvtColor') as mock_cvtcolor, \
                 patch('cv_tools.cv2.addWeighted') as mock_addWeighted, \
                 patch('cv_tools.cv2.rectangle') as mock_rectangle, \
                 patch('cv_tools.Image.fromarray') as mock_fromarray, \
                 patch('segment_anything.SamAutomaticMaskGenerator') as mock_mask_generator_class:

                # Mock SAM model
                mock_sam_model = Mock()
                mock_get_sam_model.return_value = mock_sam_model

                # Mock mask generator
                mock_mask_generator = Mock()
                mock_mask_generator_class.return_value = mock_mask_generator

                # Mock generated masks with realistic data
                mock_mask_array_1 = np.zeros((100, 100), dtype=bool)
                mock_mask_array_1[10:50, 15:45] = True  # Create a rectangular mask

                mock_mask_array_2 = np.zeros((100, 100), dtype=bool)
                mock_mask_array_2[50:85, 20:45] = True  # Create another rectangular mask

                mock_masks = [
                    {
                        'segmentation': mock_mask_array_1,
                        'bbox': [10, 15, 30, 40],  # x, y, width, height
                        'area': 1200,
                        'stability_score': 0.95
                    },
                    {
                        'segmentation': mock_mask_array_2,
                        'bbox': [50, 20, 25, 35],
                        'area': 875,
                        'stability_score': 0.88
                    }
                ]
                mock_mask_generator.generate.return_value = mock_masks

                # Setup image processing mocks
                mock_frombuffer.return_value = Mock()
                mock_imdecode.return_value = Mock()

                # Create a proper numpy array for the image mock
                mock_image_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
                mock_cvtcolor.return_value = mock_image_array
                mock_cvtcolor.return_value.shape = (100, 100, 3)  # Height, Width, Channels

                # Mock cv2 functions
                mock_addWeighted.return_value = mock_image_array
                mock_rectangle.return_value = None

                mock_pil_image = Mock()
                mock_fromarray.return_value = mock_pil_image
                mock_pil_image.save = Mock()

                # Call the function
                result = await cv_tools.segment_anything(
                    self.test_image_name,
                    model_type="vit_b",
                    return_masks=True
                )

                # Verify results
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["segment_count"], 2)
                self.assertEqual(result["model_used"], "SAM-VIT_B")
                self.assertIn("segmented_image_s3_key", result)
                self.assertIn("bounding_boxes", result)
                self.assertEqual(len(result["bounding_boxes"]), 2)

                # Verify bounding boxes are normalized
                for bbox in result["bounding_boxes"]:
                    self.assertIn("left", bbox)
                    self.assertIn("top", bbox)
                    self.assertIn("width", bbox)
                    self.assertIn("height", bbox)
                    self.assertIn("area", bbox)
                    self.assertIn("stability_score", bbox)

                    # Check normalization (values should be between 0 and 1)
                    self.assertGreaterEqual(bbox["left"], 0)
                    self.assertLessEqual(bbox["left"], 1)
                    self.assertGreaterEqual(bbox["top"], 0)
                    self.assertLessEqual(bbox["top"], 1)

                # Verify S3 interactions
                mock_connections.s3_client.get_object.assert_called_once()
                mock_connections.s3_client.put_object.assert_called_once()

                # Verify success message includes expected information
                self.assertIn("Successfully segmented", result["message"])
                self.assertIn("2 distinct segments", result["message"])

        asyncio.run(async_test())

    @patch('cv_tools.Connections')
    def test_segment_anything_error_handling(self, mock_connections):
        """Test SAM segmentation error handling"""
        async def async_test():
            # Setup mocks to trigger error
            mock_connections.s3_client.get_object.side_effect = Exception("S3 error")

            result = await cv_tools.segment_anything(
                "nonexistent.jpg",
                model_type="vit_b"
            )

            # Verify error handling — generic message, no raw error (H7)
            self.assertEqual(result["status"], "error")
            self.assertIn("SAM segmentation failed", result["message"])
            # Ensure raw exception is NOT leaked to caller
            self.assertNotIn("S3 error", result.get("message", ""))

        asyncio.run(async_test())


if __name__ == '__main__':
    unittest.main()
