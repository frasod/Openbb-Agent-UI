# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY comprehensive_agent ./comprehensive_agent
COPY static ./static

# Environment variables (can be overridden at runtime)
ENV SERVER_PORT=7777 \
    SERVER_HOST=0.0.0.0 \
    OLLAMA_BASE_URL=http://localhost:11434 \
    OLLAMA_MODEL=gemma2:9b \
    OPENAI_API_KEY=sk-xxx \
    LLM_PROVIDER=openai

EXPOSE 7777

CMD ["uvicorn", "comprehensive_agent.main:app", "--host", "0.0.0.0", "--port", "7777"] 