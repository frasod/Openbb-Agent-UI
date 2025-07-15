#!/bin/bash

# Comprehensive OpenBB Agent Startup Script
# This script sets up the environment and starts the agent

set -e

echo "🚀 Starting Comprehensive OpenBB Agent..."

# Add Poetry to PATH
export PATH="/home/churchill/.local/bin:$PATH"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install it first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Installing dependencies..."
poetry install --only main

echo "🔍 Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "⚠️  Warning: Ollama is not running on localhost:11434"
    echo "   Please start Ollama server: ollama serve"
    echo "   And ensure you have the model: ollama pull gemma3n:e4b"
fi

echo "🚀 Starting agent server on http://localhost:7777..."
echo "   - Use Ctrl+C to stop the server"
echo "   - Add http://localhost:7777 to OpenBB Workspace as agent endpoint"
echo ""

poetry run python -m comprehensive_agent.main