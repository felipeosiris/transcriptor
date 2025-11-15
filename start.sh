#!/bin/bash
set -e

# Railway inyecta la variable PORT automáticamente
# Si no está definida, usamos 8000 como fallback
PORT=${PORT:-8000}

echo "🚀 Starting Transcriptor API..."
echo "📡 Listening on 0.0.0.0:${PORT}"

# Iniciar uvicorn con el puerto de Railway
# IMPORTANTE: --host 0.0.0.0 permite conexiones externas (requerido por Railway)
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1