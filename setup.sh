#!/bin/bash

# Azure ML LSTM Time Series Forecasting Setup Script
echo "🚀 Setting up Azure ML LSTM Time Series Forecasting Project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or later."
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your Azure ML workspace details"
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
    poetry run pre-commit install
fi

# Run tests to verify setup
echo "🧪 Running tests..."
poetry run pytest tests/ -v

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Azure credentials"
echo "2. Run: poetry shell"
echo "3. Start with notebooks/01_setup_workspace.ipynb"
echo "4. Or run: python src/training/train_lstm.py"