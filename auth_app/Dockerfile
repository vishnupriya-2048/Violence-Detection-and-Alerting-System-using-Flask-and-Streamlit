# Use a Python 3.9 buster image for a balance of size and compatibility
FROM python:3.9-buster

# Set the working directory inside the container
WORKDIR /app

# Install minimal essential system dependencies
# These are crucial for Python development files, Git, secure connections, and MySQL client.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    git \
    ca-certificates \
    default-libmysqlclient-dev && \
    rm -rf /var/lib/apt/lists/*

# Install ONLY the necessary Python dependencies for the Flask app directly.
# mysqlclient is now included and uncommented here.
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    Flask-Login==0.6.3 \
    Flask-SQLAlchemy==3.1.1 \
    SQLAlchemy==2.0.41 \
    gunicorn==22.0.0 \
    python-dotenv==1.0.0 \
    Werkzeug==3.0.1 \
    itsdangerous==2.1.2 \
    Jinja2==3.1.2 \
    MarkupSafe==2.1.3 \
    click==8.1.7 \
    colorama==0.4.6 \
    blinker==1.7.0 \
    mysqlclient==2.2.0

# Copy the entire contents of the 'auth_app' directory (where this Dockerfile is located)
# into the /app directory in the container.
COPY auth_app/ .

# Command to run your Flask application using Gunicorn, binding to 0.0.0.0:$PORT
CMD gunicorn auth_app.app:app --bind 0.0.0.0:$PORT
