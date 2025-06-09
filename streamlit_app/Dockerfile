# Use a Python 3.9 slim image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the shared requirements.txt from the project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire contents of the 'streamlit_app' directory
# (where this Dockerfile is) into the /app directory in the container
COPY streamlit_app/ .

# Expose the port Streamlit will run on
EXPOSE 8080

# Command to run your Streamlit application
# Use specific Streamlit server arguments for Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
