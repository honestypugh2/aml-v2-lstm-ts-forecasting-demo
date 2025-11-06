# Azure ML Environment for LSTM Time Series Forecasting
FROM mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["bash"]