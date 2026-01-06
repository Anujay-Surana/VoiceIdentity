# Backend Dockerfile for Railway
FROM python:3.12-slim

WORKDIR /app

# Set cache directories so models persist in Docker image
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/pretrained_models

# Pre-download ML models during build (cached in image)
# SpeechBrain ECAPA-TDNN model (for speaker embeddings)
RUN python -c "from speechbrain.inference.speaker import EncoderClassifier; \
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', \
    savedir='/app/pretrained_models/spkrec-ecapa-voxceleb'); \
    print('SpeechBrain model cached!')"

# Whisper model (for transcription) - cache to explicit location
RUN mkdir -p /app/.cache/whisper && \
    python -c "import whisper; model = whisper.load_model('base', download_root='/app/.cache/whisper'); print('Whisper model cached!')"

# Copy application code AFTER caching models
COPY app/ ./app/

# Railway sets PORT env var dynamically
ENV PORT=8000

# Run the application - use shell form to expand $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
