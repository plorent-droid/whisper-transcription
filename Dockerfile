FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Keep yt-dlp up to date (YouTube changes its API frequently)
RUN pip install --no-cache-dir -U yt-dlp

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Gradio listens on 7860 by default
EXPOSE 7860

# Cache Whisper models in a volume-friendly location
ENV HF_HOME=/app/.cache

CMD ["python", "app.py"]
