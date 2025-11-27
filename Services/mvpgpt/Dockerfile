# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/main.py .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]