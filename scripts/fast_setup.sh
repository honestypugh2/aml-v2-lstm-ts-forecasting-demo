#!/bin/bash

# Fast setup script for Azure ML v2 LSTM Time Series Forecasting Project
# This script provides quick environment setup for development

set -e

echo "🚀 Starting fast setup for Azure ML LSTM project..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Please edit .env with your Azure ML workspace details"
fi

# Install dependencies with Poetry
echo "📦 Installing dependencies with Poetry..."
poetry install --only main

# Verify PyTorch installation
echo "🔍 Verifying PyTorch installation..."
poetry run python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully')"

# Create necessary directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed
mkdir -p outputs/models outputs/logs
mkdir -p notebooks/mlruns

# Check Azure CLI installation
if command -v az &> /dev/null; then
    echo "✅ Azure CLI is available"
    echo "💡 Run 'az login' to authenticate if not already logged in"
else
    echo "⚠️  Azure CLI not found. Install it for Azure ML workspace access:"
    echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
fi

# Display next steps
echo ""
echo "🎉 Fast setup completed successfully!"
echo ""
echo "📚 Next steps:"
echo "1. Edit .env with your Azure ML workspace configuration"
echo "2. Run: poetry shell"
echo "3. Start with notebooks/01_setup_workspace.ipynb"
echo "4. Follow the training pipeline in src/training/train_lstm.py"
echo ""
echo "🔗 Documentation: README.md"
echo "💻 Happy coding with Azure ML!"