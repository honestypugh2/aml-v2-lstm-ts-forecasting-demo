#!/bin/bash

# Azure ML LSTM Time Series Forecasting Setup Script
echo "🚀 Setting up Azure ML LSTM Time Series Forecasting Project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Install dependencies (creates .venv automatically)
echo "📦 Installing dependencies..."
uv sync

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "📝 Creating environment file..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your Azure ML workspace details"
    fi
fi

# Create directories if they don't exist
echo "📁 Creating output directories..."
mkdir -p outputs/models
mkdir -p outputs/logs
mkdir -p data/raw
mkdir -p data/processed

# Set up Git hooks (optional)
if [ -d .git ]; then
    echo "🔧 Setting up pre-commit hooks..."
    uv run pre-commit install
fi

# Run tests to verify setup
echo "🧪 Running tests..."
uv run pytest tests/ -v

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Azure credentials"
echo "2. Start with notebooks/01_setup_workspace.ipynb"
echo "3. Or run: uv run python src/training/train_lstm.py"