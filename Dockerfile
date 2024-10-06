# syntax = docker/dockerfile:experimental

# Stage 1: NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 AS cuda_base

# Stage 2: CUDA libraries cache
FROM alpine:latest AS cuda_cache
COPY --from=cuda_base /usr/local/cuda /cuda
RUN tar -czf /cuda.tar.gz -C /cuda .

# Stage 3: Python 3.11 environment
FROM python:3.11-slim-bullseye

# Copy CUDA libraries from the cache
RUN --mount=type=cache,target=/cuda_cache \
    --mount=type=bind,from=cuda_cache,source=/cuda.tar.gz,target=/cuda.tar.gz \
    mkdir -p /usr/local/cuda && \
    tar -xzf /cuda.tar.gz -C /usr/local/cuda && \
    [ -f /cuda.tar.gz ] && rm /cuda.tar.gz || echo "File /cuda.tar.gz not found, skipping removal."

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential\
    pkg-config\
    libhdf5-dev\
    python3-dev\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./

# Use the cache mount for pip and install packages
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the rest of the application files
COPY src ./src

# Set ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create a volume for datasets
VOLUME ["/app/src/.datasets"]

# Set the default command
CMD ["/bin/bash"]

# Build and run commands (commented out):
# $env:DOCKER_BUILDKIT=1; docker build -f Dockerfile -t ast_script .
# docker run -it --rm --gpus all -v C:/Users/Sidewinders/Desktop/CODE/UAV_Classification_repo:/app -w /app ast_script python3 src/script.py
# docker run -it --rm --gpus all -v C:/Users/Sidewinders/Desktop/CODE/UAV_Classification_repo:/app -w /app ast_script
# docker run -it --rm --gpus all -v C:/Users/Sidewinders/Desktop/CODE/UAV_Classification_repo/src/.datasets:/app/src/.datasets -w /app ast_script