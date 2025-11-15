FROM python:3.11-slim

# ffmpeg para soportar mp3/mp4/ogg/wav
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/

# Modelo por defecto (cámbialo si quieres más precisión: tiny/base/small/medium/large-v3)
ENV WHISPER_MODEL=base
ENV COMPUTE_TYPE=int8

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]