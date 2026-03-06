FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies (no dev deps in production)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Install Playwright browsers
RUN playwright install chromium --with-deps

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p data/chroma_db data/repos data/raw_cache

EXPOSE 7860

CMD ["python", "-m", "embodiedmind.ui.gradio_app"]
