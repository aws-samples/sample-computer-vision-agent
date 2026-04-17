"""
Test configuration and utilities for multi-agent orchestration system tests.
"""

import unittest
from unittest.mock import Mock

# Test configuration
TEST_CONFIG = {
    "mock_responses": {
        "web_search": "Web search results for query",
        "arxiv_search": "Arxiv research findings",
        "pubmed_search": "PubMed medical literature results",
        "chembl_search": "ChEMBL compound information",
        "clinicaltrials_search": "Clinical trials data",
        # CV-specific responses
        "image_description": "The image shows a person walking with a dog in a park",
        "video_analysis": "Video contains a package theft incident with clear suspect activity",
        "label_detection": ["Person", "Dog", "Package", "Vehicle"],
        "background_removal": "Background successfully removed from image",
        "crop_result": "Image successfully cropped to specified bounding box",
    },
    "test_queries": {
        "simple": "What is HER2?",
        "complex": "Research HER2 protein for drug discovery including compounds and clinical trials",
        "invalid": "",
        "long": "A" * 1000,
        # CV-specific queries
        "cv_simple": "Analyze this image",
        "cv_complex": (
            "Crop the person from this image, remove the background, "
            "and analyze the video for suspicious activities"
        ),
        "cv_video": "Analyze this package theft video for identifying features",
    },
    "test_media": {
        "test_image": "test_image.jpg",
        "test_video": "test_package_theft.mp4",
        "s3_bucket": "test-cv-bucket",
        "s3_keys": {
            "image": "mcp/test_image.jpg",
            "video": "mcp/test_package_theft.mp4",
            "cropped": "mcp/cropped_test_123456.jpg",
            "bg_removed": "mcp/test_bg_removed_123456.png"
        }
    },
}


def create_mock_mcp_client():
    """Create a mock MCP client for testing"""
    mock_client = Mock()
    mock_client.list_tools_sync.return_value = [
        Mock(name="search_tool"),
        Mock(name="query_tool"),
    ]
    return mock_client


def create_mock_streamlit():
    """Create a mock Streamlit object for testing"""
    mock_st = Mock()
    mock_st.empty.return_value = Mock()
    return mock_st


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup for all tests"""

    def setUp(self):
        """Common setup for all tests"""
        self.test_config = TEST_CONFIG
        self.mock_client = create_mock_mcp_client()
        self.mock_st = create_mock_streamlit()

    def assertContainsError(self, result, error_type=None):
        """Assert that result contains an error message"""
        self.assertIsInstance(result, str)
        self.assertIn("Error", result)
        if error_type:
            self.assertIn(error_type, result)
