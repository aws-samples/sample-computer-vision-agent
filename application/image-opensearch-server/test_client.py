"""Test module for MCP server client functionality.

This module tests the MCP (Model Context Protocol) server by sending requests
and validating responses, including initialization and tool listing.
"""

import subprocess
import json
import logging
import uuid
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("test_client")

def test_server():
    """Test the server connection and functionality."""
    # Use the current Python executable
    python_path = sys.executable
    logger.info(f"Using Python at: {python_path}")

    # Start the server process
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        [python_path, "-m", "src.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={
            "AWS_REGION": "us-east-1",
            "OPENSEARCH_ENDPOINT": "https://77hjf28lmy33bg7d2k1c.us-east-1.aoss.amazonaws.com"
        }
    )

    # Step 1: Send initialize request
    initialize_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0",
            "capabilities": {
                "tools": {
                    "execution": True
                }
            },
            "clientInfo": {
                "name": "mcp-test-client",
                "version": "1.0.0"
            }
        },
        "id": str(uuid.uuid4())
    }

    logger.info("Sending initialize request...")
    process.stdin.write(json.dumps(initialize_request) + "\n")
    process.stdin.flush()

    # Read initialize response
    init_response = process.stdout.readline()
    logger.info(f"Initialize response: {init_response}")

    # Step 2: Send initialized notification (required by the protocol)
    initialized_notification = {
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {}
    }

    logger.info("Sending initialized notification...")
    process.stdin.write(json.dumps(initialized_notification) + "\n")
    process.stdin.flush()

    # Give server time to process initialization
    time.sleep(1)

    # Step 3: List tools request
    list_tools_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": str(uuid.uuid4())
    }

    logger.info("Sending tools/list request...")
    process.stdin.write(json.dumps(list_tools_request) + "\n")
    process.stdin.flush()

    # Read list tools response
    tools_response = process.stdout.readline()
    logger.info(f"Tools response: {tools_response}")

    # Try to parse the JSON response
    try:
        tools_data = json.loads(tools_response)
        if "result" in tools_data:
            logger.info("Available tools:")
            for tool in tools_data["result"].get("tools", []):
                logger.info(f"- {tool.get('name')}: {tool.get('description')}")
    except json.JSONDecodeError:
        logger.warning("Could not parse tools response as JSON")

    # Clean up
    process.terminate()

if __name__ == "__main__":
    test_server()
