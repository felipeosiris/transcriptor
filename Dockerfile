FROM python:3.11-slim

# ffmpeg (decodificar audio) + libgomp1 (runtime OpenMP requerido por faster-whisper/ctranslate2)
RUN apt-get update && apt-get install -y ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/

# Modelo por defecto (puedes sobreescribir en Railway Variables)
ENV WHISPER_MODEL=tiny
ENV COMPUTE_TYPE=int8

EXPOSE 8000
# MUY IMPORTANTE: usar el puerto que asigne Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
