# Use official Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and .env file
COPY app.py .
COPY .env .

# Expose port 5000
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]