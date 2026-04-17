"""Utility functions for Image OpenSearch MCP Server."""

import logging
import os
import sys
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("image_opensearch_util")

def get_bedrock_client(region=None):
    """Get a Bedrock runtime client.

    Allows access to Bedrock models for image description and embedding generation.

    Returns:
        boto3.client: A boto3 Bedrock runtime client instance.
    """
    AWS_REGION = region or os.environ.get('AWS_REGION', 'us-east-1')

    # Configure client with timeouts
    client_config = boto3.session.Config(
        connect_timeout=5,  # 5 seconds
        read_timeout=30,    # 30 seconds
        retries={'max_attempts': 2}
    )

    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=AWS_REGION,
        config=client_config
    )
    return bedrock_client


def get_opensearch_client(region=None):
    """Get an OpenSearch client with better error handling."""
    try:
        # Clean up the endpoint URL if needed
        OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', '')
        if OPENSEARCH_ENDPOINT.startswith('https://'):
            OPENSEARCH_ENDPOINT = OPENSEARCH_ENDPOINT.replace('https://', '')

        AWS_REGION = region or os.environ.get('AWS_REGION', 'us-east-1')

        # SEC: Credential presence logging removed to prevent sensitive info disclosure
        logger.info(f"Connecting to OpenSearch at {OPENSEARCH_ENDPOINT} in {AWS_REGION}")

        # Get AWS credentials
        session = boto3.Session(region_name=AWS_REGION)
        credentials = session.get_credentials()

        if not credentials:
            logger.error("No AWS credentials found")
            return None

        # Create AWS auth - use 'aoss' for OpenSearch Serverless
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            AWS_REGION,
            'aoss',  # Important: use 'aoss' for OpenSearch Serverless
            session_token=credentials.token
        )

        logger.debug("Creating OpenSearch client...")
        opensearch_client = OpenSearch(
            hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        return opensearch_client

    except Exception as e:
        logger.error(f"Error creating OpenSearch client: {e}", exc_info=True)
        return None
