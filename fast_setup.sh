#!/bin/bash
# Fast ML Environment Setup

echo "🚀 Setting up ML environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Install all dependencies (creates .venv automatically)
echo "📦 Installing dependencies..."
uv sync

echo "✅ Environment setup complete!"
echo "💡 Run commands with: uv run <command>"