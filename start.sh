#!/bin/bash
set -e

# Obtener el puerto de Railway o usar 8000 por defecto
PORT=${PORT:-8000}

echo "Starting Transcriptor API on port $PORT"

# Iniciar uvicorn
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
