FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .
COPY start_flask_sentiment_api.py .

# Copy the model files
COPY model_files/ ./model_files/

# Set environment variables
ENV MODEL_DIR=model_files
ENV DATA_DIR=model_files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "start_flask_sentiment_api.py"]