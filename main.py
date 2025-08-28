import os, re, tempfile
from collections import defaultdict
from unidecode import unidecode
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# === Config por entorno ===
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")     # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")    # int8 en CPU

app = FastAPI(title="Transcriptor", version="1.2.1")
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

# ====== utilidades order_free (sin catálogo) ======
NUM_ES = {
    "un": 1, "uno": 1, "una": 1, "otra": 1, "otro": 1, "otros": 1, "otras": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12
}
FILLERS = {
    "de", "la", "el", "las", "los", "por", "favor", "porfa", "porfis",
    "por", "fa", "gracias", "si", "no", "me", "pon", "ponme", "pones",
    "quiero", "dame", "das", "y"
}
RE_NUM = re.compile(r"\b(\d+|{})\b".format("|".join(NUM_ES.keys())), re.IGNORECASE)

def _norm(s: str) -> str:
    import re as _re
    s = unidecode(s.strip().lower())
    s = _re.sub(r"\s+", " ", s)
    return s

def _singularize_es(token: str) -> str:
    if token.endswith("es") and len(token) > 3 and token[-3] not in "aeiou":
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token

def _clean_product_phrase(words):
    ACTIONS = {"mejor", "quitar", "quita", "cancelar", "cancela", "agrega", "añade", "pon", "ponme"}
    out = []
    for w in words:
        lw = _norm(w)
        if not lw:
            continue
        if lw in ACTIONS or lw in FILLERS:
            continue
        if lw.isdigit() or lw in NUM_ES:
            continue
        out.append(lw)
    out = [_singularize_es(w) for w in out]
    import re as _re
    phrase = " ".join(out).strip()
    phrase = _re.sub(r"\s+", " ", phrase)
    return phrase

def _word_to_int(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return NUM_ES.get(token)

def parse_order_free(text: str) -> dict:
    t = _norm(text)
    parts = [p.strip() for p in re.split(r",| y ", t) if p.strip()]
    from collections import defaultdict as _dd
    counts = _dd(int)
    last_prod = None

    for frag in parts:
        words = frag.split()
        if not words:
            continue

        is_better = words[0] == "mejor"
        is_remove = words[0] in {"quitar", "quita", "cancelar", "cancela"}

        qty = 1
        m = RE_NUM.search(frag)
        if m:
            qty = _word_to_int(m.group(1)) or 1

        product_phrase = _clean_product_phrase(words)
        if not product_phrase and any(w in {"otra", "otro", "otras", "otros"} for w in words):
            product_phrase = last_prod
        if not product_phrase and m:
            tail = frag[m.end():].strip()
            product_phrase = _clean_product_phrase(tail.split())
        if not product_phrase:
            continue

        key = product_phrase
        current = counts[key]

        if is_better:
            counts[key] = qty
        elif is_remove:
            counts[key] = max(0, current - qty) if m else 0
        else:
            counts[key] = current + qty

        last_prod = key

    products = [{k: v} for k, v in counts.items() if v > 0]
    return {"data": {"products": products}}

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
    language: str | None = Form(None),
    mode: str | None = Form(None)  # None | "order_free"
):
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

        if mode == "order_free":
            structured = parse_order_free(text)
            return JSONResponse({
                **structured,
                "raw_text": text,
                "language": info.language,
                "duration": info.duration
            })

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
