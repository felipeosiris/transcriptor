import os
import tempfile
import uuid
import yt_dlp
import ffmpeg
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# === Config por entorno ===
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")     # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")    # int8 en CPU

app = FastAPI(title="Transcriptor", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# CARGA PEREZOSA: no importes/cargues el modelo todavía
model = None
load_error = None

def get_model():
    global model, load_error
    if model is None and load_error is None:
        try:
            # importa aquí para no romper el arranque si falla la lib nativa
            from faster_whisper import WhisperModel
            model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)
        except Exception as e:
            load_error = str(e)
    return model

# ====== endpoints ======

@app.get("/health")
def health():
    status = "ok"
    detail = {}
    if load_error:
        status = "error"
        detail["load_error"] = load_error
    elif model is None:
        status = "initializing"  # aún no hemos cargado el modelo (se cargará al primer /transcribe)
    return {
        "status": status,
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        **detail
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None)
):
    """
    Endpoint básico para transcribir audio a texto.
    Recibe un archivo de audio y devuelve el texto transcrito.
    """
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = m.transcribe(tmp_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/transcribe-video")
async def transcribe_video(
    url: str = Form(...),
    language: str | None = Form(None)
):
    """
    Endpoint para transcribir videos desde URL.
    Recibe una URL de video (YouTube, Vimeo, etc.) y devuelve el texto transcrito.
    """
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )

    # Generar nombres únicos para archivos temporales
    video_id = str(uuid.uuid4())
    video_path = f"/tmp/video_{video_id}.mp4"
    audio_path = f"/tmp/audio_{video_id}.wav"

    try:
        # Configurar yt-dlp para descargar el video con múltiples opciones de formato
        ydl_opts = {
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Intentar diferentes formatos en orden de preferencia
            format_options = [
                'best[height<=720]/best[height<=480]/best',  # Preferir 720p, luego 480p, luego el mejor disponible
                'best',  # Cualquier formato disponible
                'worst',  # Como último recurso, el peor formato
            ]
            
            download_success = False
            for format_selector in format_options:
                try:
                    ydl_opts['format'] = format_selector
                    ydl = yt_dlp.YoutubeDL(ydl_opts)
                    ydl.download([url])
                    download_success = True
                    break
                except Exception as e:
                    if "Requested format is not available" in str(e):
                        continue  # Intentar siguiente formato
                    else:
                        raise e
            
            if not download_success:
                raise yt_dlp.DownloadError("No se pudo descargar el video con ningún formato disponible")

        # Verificar que el video se descargó correctamente
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail="No se pudo descargar el video")

        # Extraer audio del video usando ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )

        # Verificar que el audio se extrajo correctamente
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="No se pudo extraer el audio del video")

        # Transcribir el audio
        segments, info = m.transcribe(audio_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "video_url": url
        })

    except yt_dlp.DownloadError as e:
        return JSONResponse(
            {"error": f"Error descargando video: {str(e)}"}, 
            status_code=400
        )
    except ffmpeg.Error as e:
        return JSONResponse(
            {"error": f"Error procesando video: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Limpiar archivos temporales
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

@app.post("/transcribe-video-subtitles")
async def transcribe_video_subtitles(
    url: str = Form(...),
    language: str | None = Form(None),
    format: str = Form("json")  # json, srt, vtt
):
    """
    Endpoint para transcribir videos desde URL y devolver subtítulos con timestamps.
    Devuelve los segmentos con sus tiempos de inicio y fin para crear subtítulos.
    """
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )

    # Generar nombres únicos para archivos temporales
    video_id = str(uuid.uuid4())
    video_path = f"/tmp/video_{video_id}.mp4"
    audio_path = f"/tmp/audio_{video_id}.wav"

    try:
        # Configurar yt-dlp para descargar el video con múltiples opciones de formato
        ydl_opts = {
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Intentar diferentes formatos en orden de preferencia
            format_options = [
                'best[height<=720]/best[height<=480]/best',  # Preferir 720p, luego 480p, luego el mejor disponible
                'best',  # Cualquier formato disponible
                'worst',  # Como último recurso, el peor formato
            ]
            
            download_success = False
            for format_selector in format_options:
                try:
                    ydl_opts['format'] = format_selector
                    ydl = yt_dlp.YoutubeDL(ydl_opts)
                    ydl.download([url])
                    download_success = True
                    break
                except Exception as e:
                    if "Requested format is not available" in str(e):
                        continue  # Intentar siguiente formato
                    else:
                        raise e
            
            if not download_success:
                raise yt_dlp.DownloadError("No se pudo descargar el video con ningún formato disponible")

        # Verificar que el video se descargó correctamente
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail="No se pudo descargar el video")

        # Extraer audio del video usando ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )

        # Verificar que el audio se extrajo correctamente
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="No se pudo extraer el audio del video")

        # Transcribir el audio con segmentos detallados
        segments, info = m.transcribe(audio_path, language=language, vad_filter=True, word_timestamps=True)
        
        # Procesar segmentos para subtítulos
        subtitle_segments = []
        for i, seg in enumerate(segments):
            # Convertir palabras a diccionarios serializables
            words_data = []
            if hasattr(seg, 'words') and seg.words:
                for word in seg.words:
                    words_data.append({
                        "word": word.word,
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "probability": round(word.probability, 3) if hasattr(word, 'probability') else None
                    })
            
            subtitle_segments.append({
                "id": i + 1,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(seg.end - seg.start, 2),
                "text": seg.text.strip(),
                "words": words_data
            })

        # Formatear según el tipo solicitado
        if format.lower() == "srt":
            srt_content = format_srt(subtitle_segments)
            return JSONResponse({
                "format": "srt",
                "content": srt_content,
                "segments": subtitle_segments,
                "video_url": url,
                "language": info.language,
                "total_duration": info.duration
            })
        elif format.lower() == "vtt":
            vtt_content = format_vtt(subtitle_segments)
            return JSONResponse({
                "format": "vtt", 
                "content": vtt_content,
                "segments": subtitle_segments,
                "video_url": url,
                "language": info.language,
                "total_duration": info.duration
            })
        else:  # json por defecto
            return JSONResponse({
                "format": "json",
                "segments": subtitle_segments,
                "video_url": url,
                "language": info.language,
                "total_duration": info.duration
            })

    except yt_dlp.DownloadError as e:
        return JSONResponse(
            {"error": f"Error descargando video: {str(e)}"}, 
            status_code=400
        )
    except ffmpeg.Error as e:
        return JSONResponse(
            {"error": f"Error procesando video: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Limpiar archivos temporales
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

@app.post("/transcribe-stream")
async def transcribe_stream(
    url: str = Form(...),
    language: str | None = Form(None)
):
    """
    Endpoint específico para transcribir streams de video en vivo o URLs de streaming.
    Más flexible que el endpoint normal de video.
    """
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )

    # Generar nombres únicos para archivos temporales
    video_id = str(uuid.uuid4())
    video_path = f"/tmp/stream_{video_id}.%(ext)s"  # yt-dlp determinará la extensión
    audio_path = f"/tmp/audio_{video_id}.wav"

    try:
        # Configuración más flexible para streams con límites
        ydl_opts = {
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'format': 'best',  # Tomar el mejor formato disponible
            'prefer_insecure': True,  # Para algunos streams
            'no_check_certificate': True,  # Para streams con certificados problemáticos
            'max_filesize': 500 * 1024 * 1024,  # Límite de 500MB
            'max_downloads': 1,  # Solo un archivo
            'socket_timeout': 30,  # Timeout de 30 segundos
            'retries': 3,  # Reintentar 3 veces
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Obtener información del video primero
            info = ydl.extract_info(url, download=False)
            
            # Descargar el video
            ydl.download([url])
            
            # Buscar el archivo descargado (puede tener diferentes extensiones)
            downloaded_file = None
            for ext in ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv']:
                potential_file = f"/tmp/stream_{video_id}.{ext}"
                if os.path.exists(potential_file):
                    downloaded_file = potential_file
                    break
            
            if not downloaded_file:
                raise HTTPException(status_code=400, detail="No se pudo descargar el stream")

        # Extraer audio del video usando ffmpeg
        (
            ffmpeg
            .input(downloaded_file)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )

        # Verificar que el audio se extrajo correctamente
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="No se pudo extraer el audio del stream")

        # Transcribir el audio
        segments, info = m.transcribe(audio_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "stream_url": url,
            "type": "stream"
        })

    except yt_dlp.DownloadError as e:
        return JSONResponse(
            {"error": f"Error descargando stream: {str(e)}"}, 
            status_code=400
        )
    except ffmpeg.Error as e:
        return JSONResponse(
            {"error": f"Error procesando stream: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Limpiar archivos temporales
        try:
            # Buscar y eliminar el archivo de video descargado
            for ext in ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv']:
                potential_file = f"/tmp/stream_{video_id}.{ext}"
                if os.path.exists(potential_file):
                    os.remove(potential_file)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

@app.post("/transcribe-stream-short")
async def transcribe_stream_short(
    url: str = Form(...),
    language: str | None = Form(None),
    max_duration: int = Form(300)  # 5 minutos por defecto
):
    """
    Endpoint para streams cortos con límites estrictos.
    Ideal para clips, trailers, o videos cortos.
    """
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )

    # Generar nombres únicos para archivos temporales
    video_id = str(uuid.uuid4())
    video_path = f"/tmp/short_{video_id}.%(ext)s"
    audio_path = f"/tmp/audio_{video_id}.wav"

    try:
        # Configuración estricta para streams cortos
        ydl_opts = {
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'format': 'worst[height<=360]',  # Calidad baja para procesamiento rápido
            'max_filesize': 100 * 1024 * 1024,  # Límite de 100MB
            'max_downloads': 1,
            'socket_timeout': 15,  # Timeout corto
            'retries': 2,
            'playlistend': 1,  # Solo el primer video si es playlist
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Obtener información del video primero
            info = ydl.extract_info(url, download=False)
            
            # Verificar duración si está disponible
            if 'duration' in info and info['duration'] and info['duration'] > max_duration:
                raise HTTPException(
                    status_code=400, 
                    detail=f"El video es demasiado largo ({info['duration']}s). Máximo permitido: {max_duration}s"
                )
            
            # Descargar el video
            ydl.download([url])
            
            # Buscar el archivo descargado
            downloaded_file = None
            for ext in ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv']:
                potential_file = f"/tmp/short_{video_id}.{ext}"
                if os.path.exists(potential_file):
                    downloaded_file = potential_file
                    break
            
            if not downloaded_file:
                raise HTTPException(status_code=400, detail="No se pudo descargar el stream")

        # Extraer audio del video usando ffmpeg
        (
            ffmpeg
            .input(downloaded_file)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )

        # Verificar que el audio se extrajo correctamente
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="No se pudo extraer el audio del stream")

        # Transcribir el audio
        segments, info = m.transcribe(audio_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "stream_url": url,
            "type": "short_stream",
            "max_duration_limit": max_duration
        })

    except yt_dlp.DownloadError as e:
        return JSONResponse(
            {"error": f"Error descargando stream corto: {str(e)}"}, 
            status_code=400
        )
    except ffmpeg.Error as e:
        return JSONResponse(
            {"error": f"Error procesando stream: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Limpiar archivos temporales
        try:
            for ext in ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv']:
                potential_file = f"/tmp/short_{video_id}.{ext}"
                if os.path.exists(potential_file):
                    os.remove(potential_file)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

def format_srt(segments):
    """Formatear segmentos como SRT"""
    srt_content = ""
    for seg in segments:
        start_time = format_timestamp(seg["start"])
        end_time = format_timestamp(seg["end"])
        srt_content += f"{seg['id']}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{seg['text']}\n\n"
    return srt_content

def format_vtt(segments):
    """Formatear segmentos como VTT"""
    vtt_content = "WEBVTT\n\n"
    for seg in segments:
        start_time = format_timestamp_vtt(seg["start"])
        end_time = format_timestamp_vtt(seg["end"])
        vtt_content += f"{start_time} --> {end_time}\n"
        vtt_content += f"{seg['text']}\n\n"
    return vtt_content

def format_timestamp(seconds):
    """Formatear timestamp para SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def format_timestamp_vtt(seconds):
    """Formatear timestamp para VTT (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"