# © 2025 Amazon Web Services, Inc. or its affiliates. All Rights Reserved.
#
# This AWS Content is provided subject to the terms of the AWS Customer Agreement
# available at http://aws.amazon.com/agreement or other written agreement between
# Customer and either Amazon Web Services, Inc. or Amazon Web Services EMEA SARL or both.

"""AWS service connections module.

This module provides a centralized Connections class for managing AWS service clients
using a shared boto3 session. It handles S3, Bedrock, and Rekognition client initialization.
"""

import os
import logging
import boto3
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

class Connections:  # pylint: disable=too-few-public-methods
    """AWS service connections using a shared boto3 session."""

    logger = logger
    region_name = os.environ.get("AWS_REGION", "us-east-1")
    agent_bucket_name = os.environ.get("BUCKET_NAME", "your-bucket-name")
    logger.debug(
        f"Environment variables - AWS_REGION: {region_name}, "
        f"AGENT_BUCKET_NAME: {agent_bucket_name}"
    )
    # SEC: Credential presence logging removed to prevent sensitive info disclosure (LLM06)

    session = boto3.Session(
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )

    s3_client = session.client("s3")
    bedrock_client = session.client("bedrock-runtime")
