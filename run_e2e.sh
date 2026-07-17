#!/usr/bin/env bash
# Run Playwright end-to-end tests against the Streamlit app.
#
# Usage:
#   ./run_e2e.sh              # run all tests (headless)
#   ./run_e2e.sh --headed     # run with visible browser
#   ./run_e2e.sh --ui         # interactive UI mode
#
# Prerequisites:
#   - Node.js installed
#   - Streamlit app running at http://localhost:8501
#     (start with: streamlit run application/app.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_DIR="${SCRIPT_DIR}/e2e"

# Install deps if needed
if [ ! -d "${E2E_DIR}/node_modules" ]; then
  echo "Installing Playwright dependencies..."
  cd "${E2E_DIR}" && npm install && npx playwright install chromium
fi

cd "${E2E_DIR}"
npx playwright test "$@"
