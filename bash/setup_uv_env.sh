#!/bin/bash
# Recreate the working `crib` conda env as a uv-managed venv at .venv/.
# Uses pyproject.toml + uv.lock. Mirrors crib (Py3.9 + torch 2.1.0+cu118 + pypots 1.0 + ...).

set -e
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync --frozen

echo "--- uv env ready at .venv ---"
echo "Activate with:  source .venv/bin/activate"
