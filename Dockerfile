FROM python:3.11-slim

# ffmpeg para soportar mp3/mp4/ogg/wav + libgomp1 (requerido por faster-whisper/ctranslate2)
RUN apt-get update && apt-get install -y ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y script de inicio
COPY main.py .
COPY start.sh .

# Hacer el script ejecutable (CRÍTICO para Railway)
RUN chmod +x start.sh

# Variables de entorno
ENV WHISPER_MODEL=base
ENV COMPUTE_TYPE=int8
ENV PYTHONUNBUFFERED=1

# Exponer puerto (Railway mapeará su puerto dinámico a este)
EXPOSE 8000

# IMPORTANTE: Railway inyecta la variable PORT automáticamente
# El script start.sh la lee y la pasa a uvicorn
# --host 0.0.0.0 es CRÍTICO: permite conexiones externas (no solo localhost)
CMD ["./start.sh"]