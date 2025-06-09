# Use a Python 3.9 buster image
FROM python:3.9-buster

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by various Python packages
# This is crucial for packages like OpenCV, TensorFlow, PyTorch, SciPy, Pillow, mysqlclient etc.
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
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    libmysqlclient-dev \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the shared requirements.txt from the project root to the container's /app directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire contents of the 'auth_app' directory (where this Dockerfile is)
# into the /app directory in the container
COPY auth_app/ .

# Command to run your Flask application
CMD ["python", "app.py"]
