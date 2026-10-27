# Multi-stage Dockerfile for AutoResearcher
FROM python:3.12-slim

# Install system dependencies (build-essential, git, libgomp1 for XGBoost/CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY .env.example ./

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# Expose ports: 8501 (Streamlit Dashboard) and 5000 (MLflow UI)
EXPOSE 8501
EXPOSE 5000

# Default entrypoint starts Streamlit dashboard
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
