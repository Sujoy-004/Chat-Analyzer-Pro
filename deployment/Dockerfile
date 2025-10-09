# Chat Analyzer Pro - Docker Configuration
# Multi-stage build for optimized image size

# Stage 1: Base image with Python
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements first (for Docker layer caching)
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Stage 3: Application
FROM dependencies as application

# Create necessary directories
RUN mkdir -p data/raw data/processed notebooks src tests app

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY README.md .
COPY .streamlit/ ./.streamlit/

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative: Run as Python module
# CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py"]

# For development with hot reload:
# CMD ["streamlit", "run", "app/streamlit_app.py", "--server.fileWatcherType=poll"]
