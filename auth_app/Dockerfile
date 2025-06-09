# Use a Python 3.9 buster image
FROM python:3.9-buster

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by various Python packages
# This is a comprehensive list for scientific, ML, image processing, GIS, and database libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libxrender1 \
    libfontconfig1 \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libmysqlclient-dev \
    libsndfile1 \
    python3-dev \
    libproj-dev \   # For basemap and other GIS libraries
    libgeos-dev \   # For basemap and other GIS libraries
    ca-certificates && \ # Ensures apt-get update works securely
    rm -rf /var/lib/apt/lists/*

# Copy the shared requirements.txt from the project root to the container's /app directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire contents of the 'auth_app' directory
COPY auth_app/ .

# Command to run your Flask application
CMD ["python", "app.py"]
