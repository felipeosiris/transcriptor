# Railway Deployment Guide

## Variables de Entorno en Railway

Configura estas variables en tu proyecto de Railway:

```
WHISPER_MODEL=tiny
COMPUTE_TYPE=int8
PORT=8000
PYTHONUNBUFFERED=1
```

## Límites de Railway

- **Memoria**: 1GB (plan gratuito)
- **CPU**: Limitado
- **Tiempo de request**: 30 segundos máximo
- **Tamaño de archivo**: 200MB máximo

## Optimizaciones Implementadas

1. **Modelo pequeño**: `tiny` por defecto
2. **Timeouts**: 30 segundos para descargas
3. **Límites de archivo**: 200MB máximo
4. **Workers**: 1 solo worker
5. **Reintentos**: Máximo 2

## Endpoints Disponibles

- `GET /` - Información de la API
- `GET /health` - Estado del servicio
- `POST /transcribe` - Archivos de audio
- `POST /transcribe-video` - Videos normales
- `POST /transcribe-video-subtitles` - Videos con subtítulos
- `POST /transcribe-stream` - Streams largos
- `POST /transcribe-stream-short` - Streams cortos

## Troubleshooting

Si obtienes error 502:
1. Verifica que el modelo se cargue correctamente
2. Usa videos más cortos (< 5 minutos)
3. Usa el endpoint `/transcribe-stream-short` para videos cortos
4. Verifica los logs en Railway dashboard
