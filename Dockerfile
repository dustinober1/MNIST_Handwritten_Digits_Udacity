# Multi-stage Docker build for MNIST Digit Classifier
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mnist
USER mnist
WORKDIR /home/mnist

# Copy requirements first for better caching
COPY --chown=mnist:mnist requirements.txt .

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=mnist:mnist . .

# Add user's pip install location to PATH
ENV PATH="/home/mnist/.local/bin:${PATH}"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]