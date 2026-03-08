# Fast ML Environment Setup (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "Setting up ML environment with uv..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# Install all dependencies (creates .venv automatically)
Write-Host "Installing dependencies..." -ForegroundColor Cyan
uv sync

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "Run commands with: uv run <command>" -ForegroundColor Cyan
