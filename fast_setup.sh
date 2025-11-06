#!/bin/bash
# Fast ML Environment Setup

echo "🚀 Setting up ML environment with optimized approach..."

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch CPU version directly (much faster)
echo "📦 Installing PyTorch CPU version..."
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other ML packages with pip (faster for large packages)
echo "📦 Installing ML packages..."
pip install \
    numpy==1.24.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.4 \
    seaborn==0.12.2

# Install Azure packages
echo "☁️ Installing Azure packages..."
pip install \
    azure-ai-ml==1.30.0 \
    azure-identity==1.14.1 \
    azure-storage-blob==12.19.1 \
    mlflow==2.8.1

# Install development tools
echo "🛠️ Installing development tools..."
pip install \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    jupyterlab==4.0.9 \
    plotly==5.17.0 \
    pytest==7.4.3 \
    black==23.11.0 \
    flake8==6.1.0

echo "✅ Environment setup complete!"
echo "💡 Activate with: source .venv/bin/activate"