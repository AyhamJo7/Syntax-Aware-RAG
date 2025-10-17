FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt')"

# Create data and index directories
RUN mkdir -p /app/data /app/indexes

# Expose port for future API
EXPOSE 8000

# Default command
CMD ["synrag", "--help"]
