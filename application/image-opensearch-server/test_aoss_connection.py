"""Test module for Amazon OpenSearch Serverless (AOSS) connection.

This module tests the connection to OpenSearch Serverless, including authentication,
index operations, and basic CRUD functionality.
"""

import logging
import os
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("test_aoss_connection")

def test_connection():
    """Test the connection to the OpenSearch server."""
    # Configuration
    host = '77hjf28lmy33bg7d2k1c.us-east-1.aoss.amazonaws.com'
    region = 'us-east-1'
    service = 'aoss'  # OpenSearch Serverless
    index_name = 'test-images'

    # Get credentials
    session = boto3.Session(
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )

    credentials = session.get_credentials()
    logger.info(f"Using credentials with access key: {credentials.access_key[:4]}...{credentials.access_key[-4:]}")

    # Create auth
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

    # Create client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        retry_on_timeout=True,
        max_retries=3
    )

    # Test operations in sequence
    try:
        # Test 1: Check cluster info
        logger.info("Trying to get cluster info...")
        try:
            info = client.info()
            logger.info(f"Cluster info: {json.dumps(info, indent=2)}")
        except Exception as e:
            logger.error(f"Info failed: {e}")

        # Test 2: Check indices
        logger.info("Trying to list indices...")
        try:
            indices = client.cat.indices(v=True)  # pylint: disable=unexpected-keyword-arg
            logger.info(f"Indices: {indices}")
        except Exception as e:
            logger.error(f"List indices failed: {e}")

        # Test 3: Check if our index exists
        logger.info(f"Checking if index {index_name} exists...")
        try:
            exists = client.indices.exists(index=index_name)
            logger.info(f"Index exists: {exists}")

            # Test 4: Create index if it doesn't exist
            if not exists:
                logger.info(f"Creating index {index_name}...")
                mapping = {
                    "settings": {"index.knn": "true"},
                    "mappings": {
                        "properties": {
                            "image_vector": {
                                "type": "knn_vector",
                                "dimension": 1024
                            },
                            "photo_image_url": {"type": "keyword"},
                            "photo_description": {"type": "text"}
                        }
                    }
                }

                response = client.indices.create(
                    index=index_name,
                    body=mapping
                )
                logger.info(f"Create index response: {response}")
        except Exception as e:
            logger.error(f"Index operation failed: {e}")

    except Exception as e:
        logger.error(f"Overall test failed: {e}")

if __name__ == "__main__":
    test_connection()
