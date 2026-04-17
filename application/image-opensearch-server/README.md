# Image OpenSearch MCP Server

An MCP (Model Context Protocol) server that provides tools for generating image descriptions, creating embeddings, and ingesting/querying images in OpenSearch collections using AWS Bedrock and Amazon Titan models.

## Features

This server provides the following tools:

1. **generate_image_description** - Generate concise descriptions of images using AWS Bedrock Claude models
2. **generate_multimodal_embedding** - Create multimodal embeddings for images and text using Amazon Titan
3. **ingest_image_to_opensearch** - Generate descriptions and embeddings for images and ingest them into OpenSearch
4. **query_images_by_text** - Search for similar images using text queries
5. **query_images_by_image** - Search for similar images using image queries
6. **bulk_ingest_images** - Bulk ingest multiple images with generated descriptions and embeddings

## Prerequisites

- Python 3.10 or higher
- AWS credentials with access to Bedrock runtime
- OpenSearch cluster endpoint and credentials (if using OpenSearch features)

## Installation

The server is already installed and configured in your MCP settings. The installation includes:

- Python virtual environment with Python 3.12
- All required dependencies (mcp, boto3, opensearch-py, httpx, etc.)
- Proper project structure with FastMCP framework

## Configuration

The server is configured via environment variables in the MCP settings:

### Required AWS Variables
- `AWS_ACCESS_KEY_ID` - Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret access key  
- `AWS_REGION` - AWS region (default: us-east-1)


## Setup for Claude Desktop (Client)

### 1. Create AWS Profile

Create a named AWS profile with your credentials:

```bash
aws configure --profile mcp-temp-profile
```

Enter your AWS credentials, default region, and output format when prompted.

### 2. Configure for MCP Clients:

#### 2.1 Claude Desktop

Update your `claude_desktop_config.json` file to include the MCP server configuration:

```json
{
  "mcpServers": {
    "image-opensearch-server": {
      "command": "/path/to/image-opensearch-server/run_server.sh",
      "env": {
        "AWS_REGION": "us-east-1",
        "OPENSEARCH_ENDPOINT": "your-opensearch-endpoint.aoss.amazonaws.com",
        "AWS_PROFILE": "mcp-temp-profile"
      }
    }
  }
}
```

Important settings to update:
- `command`: Path to your run_server.sh script
- `AWS_REGION`: Your AWS region
- `OPENSEARCH_ENDPOINT`: Your OpenSearch endpoint URL
- `AWS_PROFILE`: The AWS profile name you created

#### 2.1 Amazon Q CLI
Update your `~/.aws/amazonq/mcp.json`, by adding

```json
    "image-opensearch-server": {
      "command": "/path/to/image-opensearch-server/run_server.sh",
      "args": [],
      "env": {
        "AWS_REGION": "us-east-1",
        "OPENSEARCH_ENDPOINT": "your-opensearch-endpoint.aoss.amazonaws.com",
        "AWS_PROFILE": "mcp-temp-profile"
      },
      "timeout": 120000
    }
```

### 3. Ensure Proper AWS Permissions

For OpenSearch Serverless, make sure your AWS user has:
- Access to Bedrock models
- A collection access policy for your OpenSearch collection
- Permissions to write to the collection

### 4. Start Claude Desktop

After saving the configuration:
1. Restart Claude Desktop to apply the changes
2. The MCP server will automatically start when Claude Desktop launches
3. Test the server by making a request like: "Generate a description for this image: [URL]"

### Troubleshooting Connection Issues

If you see 500 errors when accessing OpenSearch:
1. Verify your AWS profile credentials are valid: `aws sts get-caller-identity --profile mcp-temp-profile`
2. Check that your AWS user has permission to access the OpenSearch collection
3. Ensure the OpenSearch endpoint is correct and accessible
4. Look at the MCP server logs for detailed error messages

## Usage Examples

Once configured, you can use the tools like this:

### Generate Image Description
```
Generate a description for this image: https://example.com/image.jpg
```

### Ingest Image to OpenSearch
```
Ingest this image into OpenSearch index "imgs-vector-index-test": https://example.com/image.jpg
```

### Search Images by Text
```
Search for images containing "sunset over mountains" in index "imgs-vector-index-test"
```

### Search Images by Image
```
Find similar images to https://example.com/query-image.jpg in index "imgs-vector-index-test"
```

### Bulk Ingest Images
```
Bulk ingest these images into index "imgs-vector-index-test": ["url1.jpg", "url2.jpg", "url3.jpg"]
```

## Supported Models

- **Image Description**: Anthropic Claude 3 Sonnet (default: anthropic.claude-3-sonnet-20240229-v1:0)
- **Embeddings**: Amazon Titan Multimodal Embeddings (amazon.titan-embed-image-v1)

## Architecture

The server is built using:
- **FastMCP**: Modern MCP server framework
- **AWS Bedrock**: For image description and embedding generation
- **OpenSearch**: For vector storage and similarity search
- **AsyncIO**: For efficient async operations

## File Structure

```
image-opensearch-server/
├── src/
│   ├── __init__.py
│   ├── server.py      # Main MCP server implementation
│   └── util.py        # AWS and OpenSearch client utilities
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── venv/            # Python virtual environment
```

## Error Handling

The server includes comprehensive error handling:
- AWS service errors (authentication, model access, etc.)
- OpenSearch connection and indexing errors
- Image download and processing errors
- Input validation errors

## Security Notes

- The server uses the AWS credentials configured in MCP settings
- OpenSearch connections use SSL by default
- Certificate verification is disabled for development (should be enabled in production)
- Credentials are passed via environment variables, not stored in code

## Troubleshooting

1. **Server not connecting**: Check AWS credentials and region settings
2. **OpenSearch errors**: Verify endpoint URL and authentication credentials
3. **Model access errors**: Ensure Bedrock model access is enabled in your AWS account
4. **Image download errors**: Verify image URLs are accessible and valid

## Development

To modify or extend the server:

1. Navigate to the server directory:
   ```bash
   cd image-opensearch-server
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Make your changes to the source files

4. Test the server:
   ```bash
   python -m src.server --help
   ```

The server will automatically reload when MCP detects changes.