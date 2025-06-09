# Use a Python 3.9 slim image for a smaller footprint
FROM python:3.9-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the shared requirements.txt from the project root to the container's /app directory
# (This assumes requirements.txt is in your_project_root/ )
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire contents of the 'auth_app' directory (where this Dockerfile is)
# into the /app directory in the container
COPY auth_app/ .

# Command to run your Flask application
CMD ["python", "app.py"]
