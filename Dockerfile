# Base image with PyTorch 2.0+ and CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Metadata
LABEL maintainer="Antigravity Agent"
LABEL description="Ternary VAEs Bioinformatics Environment"

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for bio-tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command
CMD ["pytest"]
