FROM python:3.11-slim

# ffmpeg (decodificar audio) + libgomp1 (runtime OpenMP requerido por faster-whisper/ctranslate2)
RUN apt-get update && apt-get install -y ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY main.py .

# Variables de entorno
ENV WHISPER_MODEL=tiny
ENV COMPUTE_TYPE=int8
ENV PYTHONUNBUFFERED=1

# Exponer puerto (Railway mapeará su puerto dinámico a este)
EXPOSE 8000

# Comando de inicio usando la variable PORT de Railway
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"

