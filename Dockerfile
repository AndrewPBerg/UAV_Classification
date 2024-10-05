# syntax = docker/dockerfile:experimental

# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

# Set the working directory
WORKDIR /home/appuser

# Copy the temporary requirements file
COPY requirements.txt /app/requirements.txt

# Use the cache mount for pip and install packages
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip install --upgrade pip==21.3.1 && \
    pip install -r /app/requirements.txt

# Copy the rest of the application files
COPY --chown=appuser:appuser . /src/

VOLUME ["/.datasets"]

# Expose application port if necessary
# EXPOSE 8000

# Set the default command
CMD ["python3", "script.py"]


# docker build -f Dockerfile -t ast_script .
# docker run -it --rm --gpus all -v C:/Users/Sidewinders/Desktop/CODE/UAV_Classification_repo:/app -w /app ast_script python3 src/AST_Script.py

# docker run -it --rm --gpus all -v C:/Users/Sidewinders/Desktop/CODE/UAV_Classification_repo:/app -w /app ast_script