# Use a Python 3.9 slim image for a smaller image size
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for:
# - libgl1: OpenCV (cv2) in a headless environment
# - libpq-dev: PostgreSQL client libraries for psycopg2-binary
# - git, ca-certificates: General utilities and secure connections
# - build-essential, cmake: Often needed for packages with C/C++ extensions, including aiortc
# - ffmpeg: Sometimes helpful for video processing with OpenCV/av
# - libsdl2-dev, libopus-dev, libvpx-dev: Additional libraries for aiortc's WebRTC functionalities
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

# Copy the Streamlit app's specific requirements.txt
# This assumes your requirements.txt is in the streamlit_app/ directory locally
COPY streamlit_app/requirements.txt ./requirements.txt

# Install Python dependencies from the Streamlit app's requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire contents of the 'streamlit_app' directory locally
# (which contains app.py, models/, etc.) into the /app directory in the container
COPY streamlit_app/ .

# Expose the port Streamlit will run on (Render automatically uses $PORT)
EXPOSE 8080

# Command to run your Streamlit application
# Streamlit will listen on the port provided by Render ($PORT)
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
