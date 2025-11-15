FROM python:3.11-slim

# ffmpeg (decodificar audio) + libgomp1 (runtime OpenMP requerido por faster-whisper/ctranslate2)
RUN apt-get update && apt-get install -y ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y script de inicio
COPY main.py .
COPY start.sh .

# Hacer el script ejecutable
RUN chmod +x start.sh

# Variables de entorno
ENV WHISPER_MODEL=tiny
ENV COMPUTE_TYPE=int8
ENV PYTHONUNBUFFERED=1

# Exponer puerto (Railway mapeará su puerto dinámico a este)
EXPOSE 8000

# Usar el script de inicio
CMD ["./start.sh"]

