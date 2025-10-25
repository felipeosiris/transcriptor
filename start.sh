#!/bin/bash

# Script de inicio para Railway
echo "Starting Transcriptor API..."

# Verificar que PORT esté definido
if [ -z "$PORT" ]; then
    echo "PORT not set, using default 8000"
    export PORT=8000
fi

echo "Using port: $PORT"

# Iniciar la aplicación
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 30
