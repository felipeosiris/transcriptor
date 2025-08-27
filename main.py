import os, tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")   # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # int8 es ligero para CPU

app = FastAPI(title="Transcriptor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "compute_type": COMPUTE_TYPE}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try: os.remove(tmp_path)
        except: pass
