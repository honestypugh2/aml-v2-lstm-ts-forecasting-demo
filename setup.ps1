# Azure ML LSTM Time Series Forecasting Setup Script (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "Setting up Azure ML LSTM Time Series Forecasting Project..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# Install dependencies (creates .venv automatically)
Write-Host "Installing dependencies..." -ForegroundColor Cyan
uv sync

# Create environment file if it doesn't exist
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "Creating environment file..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "Please edit .env file with your Azure ML workspace details" -ForegroundColor Yellow
    }
}

# Create directories if they don't exist
Write-Host "Creating output directories..." -ForegroundColor Cyan
$dirs = @("outputs/models", "outputs/logs", "data/raw", "data/processed")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Set up Git hooks (optional)
if (Test-Path ".git") {
    Write-Host "Setting up pre-commit hooks..." -ForegroundColor Cyan
    uv run pre-commit install
}

# Run tests to verify setup
Write-Host "Running tests..." -ForegroundColor Cyan
uv run pytest tests/ -v

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Edit .env file with your Azure credentials"
Write-Host "2. Start with notebooks/01_setup_workspace.ipynb"
