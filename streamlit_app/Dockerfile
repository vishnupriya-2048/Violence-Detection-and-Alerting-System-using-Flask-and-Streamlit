FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libpq-dev \
    git \
    ca-certificates \
    build-essential \
    cmake \
    ffmpeg \
    libsdl2-dev \
    libopus-dev \
    libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt (now in same folder as Dockerfile)
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the streamlit application file
COPY streamlit_app.py ./app.py

# Create models directory and copy YOLO models
RUN mkdir -p models
COPY VoilenceDetection/best.pt ./models/best.pt
COPY yolov8n.pt ./models/yolov8n.pt

# Expose port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
