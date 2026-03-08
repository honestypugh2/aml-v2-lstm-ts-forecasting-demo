# Fast setup script for Azure ML v2 LSTM Time Series Forecasting Project (PowerShell)
# This script provides quick environment setup for development
$ErrorActionPreference = "Stop"

Write-Host "Starting fast setup for Azure ML LSTM project..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Please install uv first:" -ForegroundColor Red
    Write-Host "   powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "Creating .env file from template..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "Please edit .env with your Azure ML workspace details" -ForegroundColor Green
    }
}

# Install dependencies with uv (main only)
Write-Host "Installing dependencies with uv..." -ForegroundColor Cyan
uv sync --no-dev

# Verify PyTorch installation
Write-Host "Verifying PyTorch installation..." -ForegroundColor Cyan
uv run python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# Create necessary directories if they don't exist
Write-Host "Creating necessary directories..." -ForegroundColor Cyan
$dirs = @("data/raw", "data/processed", "outputs/models", "outputs/logs", "notebooks/mlruns")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Check Azure CLI installation
if (Get-Command az -ErrorAction SilentlyContinue) {
    Write-Host "Azure CLI is available" -ForegroundColor Green
    Write-Host "Run 'az login' to authenticate if not already logged in" -ForegroundColor Cyan
} else {
    Write-Host "Azure CLI not found. Install it for Azure ML workspace access:" -ForegroundColor Yellow
    Write-Host "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
}

# Display next steps
Write-Host ""
Write-Host "Fast setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Edit .env with your Azure ML workspace configuration"
Write-Host "2. Start with notebooks/01_setup_workspace.ipynb"
Write-Host "3. Follow the training pipeline in src/training/train_lstm.py"
Write-Host ""
Write-Host "Documentation: README.md"
