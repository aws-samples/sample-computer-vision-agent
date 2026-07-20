# © 2025 Amazon Web Services, Inc. or its affiliates. All Rights Reserved.
#
# This AWS Content is provided subject to the terms of the AWS Customer Agreement
# available at http://aws.amazon.com/agreement or other written agreement between
# Customer and either Amazon Web Services, Inc. or Amazon Web Services EMEA SARL or both.

"""AWS Bedrock utilities module.

This module provides utility functions for interacting with AWS Bedrock models,
including creating multimodal prompts and invoking Bedrock models for image analysis.
"""

import json
import base64
from typing import Dict, Any

from application.aws_cv_mcp_server.connections import Connections

logger = Connections.logger


def invoke_bedrock_model(  # pylint: disable=unused-argument
    prompt: Dict[str, Any],
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    max_tokens: int = 1000,
    temperature: float = 0.5,
) -> str:
    """
    Invoke Bedrock model with given prompt and return the response text.

    Args:
        prompt: Dictionary containing the prompt structure
        model_id: Bedrock model ID to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for response generation

    Returns:
        str: Model's response text
    """
    try:
        # logger.info(f"Prompt for Bedrock: {prompt}")

        response = Connections.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(prompt),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response.get("body").read())
        # logger.info(f"Bedrock response: {response_body}")

        analysis = response_body["content"][0]["text"]
        # logger.info(f"Bedrock analysis: {analysis}")

        return analysis

    except Exception as e:
        logger.error(f"Error invoking Bedrock: {e}")
        raise


def create_text_prompt(
    content: str,
    system_prompt: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.5,
) -> Dict[str, Any]:
    """Create a text-only prompt structure"""
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    if system_prompt:
        prompt["system"] = system_prompt
    return prompt


def create_multimodal_prompt(  # pylint: disable=too-many-positional-arguments
    image_data: bytes,
    text: str,
    content_type: str,
    system_prompt: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.5,
) -> Dict[str, Any]:
    """Create a multimodal prompt structure with image and text"""
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": base64.b64encode(image_data).decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
    }
    if system_prompt:
        prompt["system"] = system_prompt
    return prompt
