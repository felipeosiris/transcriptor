import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
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
        "version": "2.0.0",
        **detail
    }

@app.get("/")
def root():
    return {
        "message": "Transcriptor API",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/transcribe"
        ]
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None)
):
    """
    Endpoint para transcribir audio a texto.
    Recibe un archivo de audio en cualquier formato y devuelve el texto transcrito.
    """
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )

    tmp_path = None
    try:
        # Leer el archivo
        file_content = await file.read()
        if not file_content:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # Transcribir
        segments, info = m.transcribe(tmp_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        return JSONResponse({
            "error": error_msg,
            "traceback": traceback_str
        }, status_code=500)
    finally:
        # Limpiar archivo temporal
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass