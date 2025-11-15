FROM python:3.11-slim

# ffmpeg para soportar mp3/mp4/ogg/wav + libgomp1 (requerido por faster-whisper/ctranslate2)
RUN apt-get update && apt-get install -y ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY main.py .

# Variables de entorno
ENV WHISPER_MODEL=base
ENV COMPUTE_TYPE=int8
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Usar variable PORT de Railway (con fallback a 8000)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"