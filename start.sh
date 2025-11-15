#!/usr/bin/env bash
set -e

# Railway inyecta la variable PORT automáticamente en el entorno
# Leemos PORT del entorno, con fallback a 8000 si no está definida
PORT="${PORT:-8000}"

echo "🚀 Starting Transcriptor API..."
echo "📡 Listening on 0.0.0.0:${PORT}"

# Iniciar uvicorn con configuración para Railway:
# - --host 0.0.0.0: CRÍTICO - permite conexiones externas (no solo localhost)
# - --port $PORT: Usa el puerto que Railway asigna dinámicamente
# - --workers 1: Un solo worker para Railway (evita problemas de memoria)
# - exec: Reemplaza el proceso del shell para que Railway gestione el proceso correctamente
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --workers 1