# Multi-Agent Computer Vision Framework with MCP and Strands Agents

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Infrastructure Deployment](#infrastructure-deployment)
- [Directory Structure](#directory-structure)
- [How It Works](#how-it-works)
- [MCP Servers](#mcp-servers)
- [Available Tools](#available-tools)
- [Data Flow and Routing](#data-flow-and-routing)
- [Security](#security)
- [Extension Patterns](#extension-patterns)
- [Configuration](#configuration)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Contributing](#contributing)
- [License](#license)

## Overview

A multi-agent computer vision framework that combines
[Strands Agents][strands] and
[Model Context Protocol (MCP)][mcp] with AWS services
to perform image analysis, video understanding, object
segmentation, background removal, and semantic image
search.

The system uses an orchestrator agent that delegates
tasks to specialist agents, each backed by dedicated
MCP servers exposing domain-specific tools. This
separation of concerns allows each component to scale,
evolve, and be tested independently.

Key capabilities:

- Image analysis and description via Amazon Bedrock
  (Claude, Nova)
- Object detection and label recognition via Amazon
  Rekognition
- Object segmentation via Meta's Segment Anything
  Model (SAM)
- Background removal via rembg
- Video analysis via Amazon Nova
- Semantic image search via Amazon OpenSearch
  Serverless with Titan embeddings
- Streamlit web UI with media upload and
  conversational interaction

[strands]: https://github.com/strands-agents/strands-agents-python
[mcp]: https://modelcontextprotocol.io/

## Architecture

![Architecture Diagram](docs/architecture.png)

## Quick Start

Prerequisites:

- Python 3.11+
- AWS account with access to Bedrock, S3,
  Rekognition, and (optionally) OpenSearch Serverless
- AWS credentials configured (via environment
  variables or assumed role)
- A VPC with subnets and security groups (required
  for OpenSearch Serverless VPC endpoint)

### 1. Deploy infrastructure

```bash
aws cloudformation deploy \
  --template-file cfn.yaml \
  --stack-name cv-mcp-server \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    CollectionName=collection \
    VPCId=vpc-xxxxxxxx \
    SubnetIds=subnet-aaa,subnet-bbb \
    SecurityGroupIds=sg-xxxxxxxx
```

### 2. Set up credentials

Retrieve the generated credentials from Secrets
Manager and export them:

```bash
eval "$(aws secretsmanager get-secret-value \
  --secret-id cv-mcp-server-unix-credentials \
  --query SecretString --output text)"
```

Or create a `.env` file manually:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token
AWS_REGION=us-east-1
BUCKET_NAME=your-cv-bucket
OPENSEARCH_ENDPOINT=your-endpoint.aoss.amazonaws.com
```

### 3. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run

```bash
streamlit run application/app.py
```

## Infrastructure Deployment

The `cfn.yaml` CloudFormation template provisions:

- S3 bucket with encryption at rest (AES-256),
  versioning, public access block, a bucket policy
  enforcing HTTPS-only access
  (`DenyInsecureTransport`), server access logging,
  and a lifecycle policy that expires uploaded media
  after 30 days
- S3 access logs bucket with 90-day retention for
  audit trail
- IAM role with scoped permissions for S3, Bedrock,
  Rekognition, OpenSearch, and CloudWatch Logs
- IAM user with assume-role capability for local
  development
- OpenSearch Serverless collection (vector search)
  with encryption, VPC-restricted network policy,
  and data access policies
- OpenSearch Serverless VPC endpoint for private
  network access
- Secrets Manager secrets containing ready-to-use
  credential export commands for Unix/macOS and
  Windows

The IAM policies follow least-privilege principles:

- S3 actions scoped to the specific bucket
- Bedrock `InvokeModel` scoped to specific model
  ARNs in the deployment region only, enforced via
  `aws:RequestedRegion` condition
- OpenSearch `APIAccessAll` and `UpdateCollection`
  scoped to the specific collection ARN;
  only `BatchGetCollection` and `ListCollections`
  use `Resource: "*"` (required by AWS)
- CloudWatch Logs scoped to the
  `/cv-mcp-server/*` log group prefix
- Rekognition `DetectLabels` (requires
  `Resource: "*"` per AWS documentation)

### CloudFormation parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `CollectionName` | No | OpenSearch collection name (default: `collection`) |
| `VPCId` | Yes | VPC ID for OpenSearch VPC endpoint |
| `SubnetIds` | Yes | Subnet IDs for OpenSearch VPC endpoint |
| `SecurityGroupIds` | Yes | Security group IDs for OpenSearch VPC endpoint |

## Directory Structure

```text
├── application/
│   ├── app.py                  # Streamlit UI with media upload
│   ├── chat.py                 # Multi-agent orchestration
│   ├── info.py                 # Model configuration
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── cv_agent.py         # CV specialist prompt
│   │   └── interaction_agent.py # Orchestrator prompt
│   ├── aws_cv_mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py           # CV MCP server entry
│   │   ├── cv_tools.py         # CV tool implementations
│   │   ├── bedrock_utils.py    # Bedrock invocation utils
│   │   ├── connections.py      # AWS client management
│   │   ├── models.py           # Pydantic response models
│   │   └── scanner.py          # Image scanning utils
│   └── image-opensearch-server/
│       ├── src/
│       │   ├── server.py       # OpenSearch MCP server
│       │   └── util.py         # Client factories
│       ├── test_client.py      # MCP client tests
│       ├── test_aoss_connection.py
│       ├── requirements.txt
│       └── README.md
├── tests/
│   ├── __init__.py
│   ├── run_tests.py
│   ├── test_config.py
│   ├── test_cv_tools.py
│   └── test_cv_integration.py
├── assets/
│   ├── test_image.png
│   ├── test_image_pexels.jpg
│   ├── test_video.mp4
│   └── AmazonEmber_Lt.ttf
├── docs/
│   ├── architecture.png
│   └── architecture.xml
├── cfn.yaml                    # CloudFormation template
├── requirements.txt            # Pinned Python dependencies
├── pyproject.toml
├── .pylintrc
├── .python-version
├── .env                        # Credentials (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

## How It Works

### Agent hierarchy

The framework uses a three-tier agent architecture:

1. The Interaction Agent (orchestrator) receives user
   queries, determines which specialist to invoke,
   and coordinates multi-step workflows.
2. The CV Agent handles S3-based image and video
   operations: cropping, label detection, description,
   background removal, SAM segmentation, and video
   analysis.
3. The Image OpenSearch Agent handles URL-based
   images: generating descriptions, creating
   multimodal embeddings, ingesting into OpenSearch,
   and running similarity searches.

Agents are built with [Strands Agents][strands] and
communicate with tools via MCP. Each agent's behavior
is defined by a system prompt in
`application/prompts/`.

### Thread-safe tool creation

MCP clients are opened once per request and
distributed to specialist agents via closures,
avoiding Streamlit session-state threading issues:

```python
def create_specialist_tools_with_clients(
    cv_client, opensearch_client
):
    @tool
    def cv_agent(query: str) -> str:
        return cv_agent_impl(query, cv_client)

    @tool
    def image_opensearch_agent(query: str) -> str:
        return image_opensearch_agent_impl(
            query, opensearch_client
        )

    return [cv_agent, image_opensearch_agent]
```

### Image display

Background agent threads store image references in a
thread-safe global list. After the agent response
completes, the main Streamlit thread retrieves them
for display. This avoids Streamlit's "missing
ScriptRunContext" errors.

## MCP Servers

### CV MCP Server (`aws_cv_mcp_server/`)

Runs as a subprocess via stdio transport:

```text
.venv/bin/python -m application.aws_cv_mcp_server.server
```

Exposes tools for image/video processing backed by
S3, Bedrock, and Rekognition. Includes filename
sanitization to prevent path traversal (CWE-22),
generic error messages to prevent information
disclosure (CWE-209), and rate limiting on expensive
operations.

### Image OpenSearch MCP Server (`image-opensearch-server/`)

Runs as a subprocess with its own virtual environment:

```text
python -m src.server
```

Exposes tools for image description, embedding
generation, and OpenSearch vector search. Includes
URL validation to prevent SSRF (CWE-918) — blocks
private IPs, loopback, link-local, reserved ranges,
and the AWS metadata service. HTTP redirects are
disabled. Bulk ingest operations are rate-limited.

## Available Tools

### CV Tools

| Tool | Description | AWS Service |
|------|-------------|-------------|
| `describe_image` | Analyze image content | Bedrock (Claude) |
| `detect_labels` | Detect objects with bounding boxes | Rekognition |
| `crop_bounding_box` | Extract a region from an image | S3 + Pillow |
| `remove_background` | Remove image background | rembg (ONNX) |
| `segment_anything` | Segment all objects (SAM) | SAM + PyTorch |
| `analyze_video` | Analyze video content | Bedrock (Nova) |

### OpenSearch Tools

| Tool | Description | AWS Service |
|------|-------------|-------------|
| `generate_image_description` | Describe image from URL | Bedrock (Claude) |
| `generate_multimodal_embedding` | Create vector embedding | Bedrock (Titan) |
| `ingest_image_to_opensearch` | Describe, embed, index | Bedrock + OpenSearch |
| `query_images_by_text` | Search images by text | Bedrock + OpenSearch |
| `query_images_by_image` | Find similar images | Bedrock + OpenSearch |
| `bulk_ingest_images` | Batch ingest images | Bedrock + OpenSearch |

### UI Tools

| Tool | Description |
|------|-------------|
| `ui_show_image` | Display a single image with caption |
| `ui_show_images` | Display multiple images in a grid |

## Data Flow and Routing

The orchestrator routes requests based on input type:

- Input contains `http://` or `https://` →
  Image OpenSearch Agent
- Input references uploaded/S3 files → CV Agent
- Display requests follow a two-step process:
  analyze content first, then display with
  AI-generated caption

```text
User: "Crop all people from photo.jpg"
  → Orchestrator detects S3 file → CV Agent
  → detect_labels("photo.jpg") → "Person" boxes
  → crop_bounding_box("photo.jpg", bbox)
  → ui_show_images(["cropped_person_abc123.jpg"])

User: "Index this image: https://example.com/cat.jpg"
  → Orchestrator detects URL → OpenSearch Agent
  → ingest_image_to_opensearch(url, "my-index")
  → Returns: description, embedding, document ID
```

## Security

The codebase includes the following security
hardening measures:

### Input validation

- **SSRF prevention (CWE-918):** all URL-fetching
  functions validate URLs against private IP ranges,
  loopback, link-local, reserved addresses, and the
  AWS metadata endpoint (`169.254.169.254`). HTTP
  redirects are disabled (`follow_redirects=False`).
- **Path traversal prevention (CWE-22):** S3 key
  construction uses `os.path.basename()` and regex
  sanitization to strip traversal sequences.
- **File upload validation (CWE-434):** `app.py`
  checks file magic bytes against expected
  image/video signatures before accepting uploads.
- **Video format allowlist:** video file extensions
  are validated against `ALLOWED_VIDEO_FORMATS`
  before being sent to the Bedrock API.

### Information disclosure prevention

- **Generic error messages (CWE-209):** all tool
  functions return generic error messages to callers.
  Detailed errors including S3 paths, bucket names,
  and stack traces are logged server-side only.
- **No credential logging:** AWS credentials are
  never logged, even at debug level.

### Rate limiting

- **Expensive operation throttling:** `analyze_video`
  (5 calls/min), `segment_anything` (3 calls/min),
  and `bulk_ingest_images` (3 calls/min) are
  rate-limited via token-bucket limiters to prevent
  cost abuse.

### Agent security

- **Prompt injection protection:** system prompts in
  both agents include explicit rules to reject
  embedded instructions from user content.
- **Tool consent gating:** `BYPASS_TOOL_CONSENT` is
  only enabled when the `ENVIRONMENT` variable is
  explicitly set to a non-production value. Defaults
  to production (consent required).

### Model integrity

- **SAM model hash verification:** downloaded SAM
  model weights are verified against known SHA256
  hashes before loading. Files with mismatched
  hashes are deleted and rejected.

### Infrastructure security

- **S3 transport encryption:** bucket policy denies
  all requests where `aws:SecureTransport` is
  `false`.
- **S3 access logging:** all bucket access is logged
  to a dedicated access logs bucket with 90-day
  retention.
- **S3 lifecycle policy:** uploaded media under the
  `mcp/` prefix expires after 30 days; noncurrent
  versions expire after 7 days.
- **OpenSearch VPC restriction:** the OpenSearch
  Serverless collection is accessible only via a
  VPC endpoint — public access is disabled.
- **IAM least privilege:** Bedrock access is
  restricted to the deployment region via
  `aws:RequestedRegion` condition. CloudWatch Logs
  access is scoped to `/cv-mcp-server/*`.
- **Pinned dependencies:** all Python dependencies
  in `requirements.txt` are pinned to exact versions
  to prevent supply-chain attacks.

## Extension Patterns

### Add a new CV tool

1. Implement in
   `application/aws_cv_mcp_server/cv_tools.py`:

```python
async def new_tool(param: str) -> Dict[str, Any]:
    """Tool description for agent understanding."""
    # implementation
    return {"status": "success", "result": result}
```

2. Register in
   `application/aws_cv_mcp_server/server.py`:

```python
from .cv_tools import new_tool

@mcp.tool(name='new_tool')
async def mcp_new_tool(param: str) -> Dict:
    return await new_tool(param)
```

3. Update the CV agent prompt in
   `application/prompts/cv_agent.py` to reference
   the new tool.

### Add a new specialist agent

1. Create `application/prompts/new_agent.py` with
   `SYSTEM_PROMPT`
2. Implement `new_agent_impl()` in `chat.py`
   following the `cv_agent_impl()` pattern
3. Add to `create_specialist_tools_with_clients()`
   as a new `@tool`
4. Update the orchestrator prompt in
   `interaction_agent.py` with delegation
   instructions

## Configuration

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS secret key |
| `AWS_SESSION_TOKEN` | No | Session token (assumed roles) |
| `AWS_REGION` | Yes | AWS region (default: `us-east-1`) |
| `BUCKET_NAME` | Yes | S3 bucket for media storage |
| `OPENSEARCH_ENDPOINT` | No | OpenSearch Serverless endpoint |
| `ENVIRONMENT` | No | Set to non-`production` to enable tool consent bypass |

### Model selection

Models are configured in `application/info.py` with
regional availability. The default is Claude 3.7
Sonnet. Available options in the Streamlit sidebar:

- Claude 4 Sonnet (with interleaved thinking)
- Claude 3.7 Sonnet (with extended thinking)
- Claude 3.5 Sonnet
- Claude 3.5 Haiku

Video analysis uses Amazon Nova models (Lite, Pro,
Premier).

## Testing

```bash
# Run all tests
AWS_REGION=us-east-1 BUCKET_NAME=test-bucket \
  python -m pytest tests/ -v

# Or use the test runner
python tests/run_tests.py
```

Tests mock AWS services for fast, offline execution.
Test assets are in `assets/`. The `AWS_REGION` and
`BUCKET_NAME` environment variables must be set for
test collection to succeed.

## Code Quality

```bash
python -m pylint application/ tests/ --rcfile=.pylintrc
```

Target: maintain score >= 9.5/10. The project follows
PEP 8 with project-specific adjustments defined in
`.pylintrc`.

## Monitoring and Debugging

All application modules use Python's `logging` module
with structured output to stderr:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
```

Enable debug-level logging for detailed tracing:

```python
logging.basicConfig(level=logging.DEBUG)
```

Common issues:

- MCP connection failures: check virtual environment
  paths in `chat.py` and verify AWS credentials
- Image display issues: verify S3 bucket name and
  permissions
- Model errors: confirm Bedrock model access is
  enabled in your region
- Rate limit errors: expensive operations
  (`analyze_video`, `segment_anything`,
  `bulk_ingest_images`) are throttled — wait and
  retry

## Contributing

1. Run tests:
   `python -m pytest tests/ -v`
2. Run linting:
   `python -m pylint application/ tests/ --rcfile=.pylintrc`
3. Follow PEP 8, add docstrings to public functions,
   use type hints
4. Update this README when adding features
5. Update agent prompts when changing tool behavior

## License

This project is licensed under MIT-0.
See [LICENSE](LICENSE) for details.
