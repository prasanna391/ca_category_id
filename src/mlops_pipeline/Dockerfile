# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install necessary dependencies including pandas and scipy
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container
COPY . .

# Set the environment variable to ensure Python outputs prints directly to terminal logs
ENV PYTHONUNBUFFERED=1

# Expose the internal container port to the host
EXPOSE 5000

# Run your file directly as a script so the Flask app remains alive
CMD ["python", "src/mlops_pipeline/11_drift_and_retrain.py"]
