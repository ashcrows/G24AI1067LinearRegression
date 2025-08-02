# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run predict script as container's main process
CMD ["python", "src/predict.py"]
