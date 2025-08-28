FROM python:3.11-slim

# ffmpeg para soportar mp3/mp4/ogg/wav
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/

# Variables por defecto (puedes cambiarlas en Railway Variables)
ENV WHISPER_MODEL=tiny
ENV COMPUTE_TYPE=int8

EXPOSE 8000
# Usa el puerto asignado por la plataforma si existe
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
