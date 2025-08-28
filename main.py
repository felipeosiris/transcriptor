import os, re, tempfile
from collections import defaultdict
from unidecode import unidecode
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# === Configuración de modelo (vía variables de entorno) ===
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")      # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")     # int8 recomendado en CPU (Railway/Render)

app = FastAPI(title="Transcriptor", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Carga única del modelo al iniciar el contenedor
model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)

# ========================
# Utilidades de parsing libre (sin catálogo)
# ========================

# números básicos en español
NUM_ES = {
    "un": 1, "uno": 1, "una": 1, "otra": 1, "otro": 1, "otros": 1, "otras": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12
}

# palabras de relleno comunes
FILLERS = {
    "de", "la", "el", "las", "los", "por", "favor", "porfa", "porfis",
    "por", "fa", "gracias", "si", "no", "me", "pon", "ponme", "pones",
    "quiero", "dame", "das", "y"
}

RE_NUM = re.compile(r"\b(\d+|{})\b".format("|".join(NUM_ES.keys())), re.IGNORECASE)

def _norm(s: str) -> str:
    s = unidecode(s.strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s

def _singularize_es(token: str) -> str:
    # Heurística simple plural->singular suficiente para: conchas, teleras, donas, panes
    if token.endswith("es") and len(token) > 3:
        if token[-3] not in "aeiou":
            return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token

def _clean_product_phrase(words):
    # Quita fillers, números y acciones; devuelve frase candidata de producto
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
    phrase = " ".join(out).strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase

def _word_to_int(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return NUM_ES.get(token)

def parse_order_free(text: str) -> dict:
    """
    Sin catálogo: agrupa por la frase de producto extraída del dictado.
    Reglas:
      - "N <producto>" -> suma N
      - "otra/otro <producto>" -> +1
      - "otra/otro" sin producto -> +1 al último producto
      - "mejor N <producto>" -> reemplaza cantidad por N (si no hay N, 1)
      - "quitar/cancela N <producto>" -> resta N; sin N -> elimina
    """
    t = _norm(text)
    parts = [p.strip() for p in re.split(r",| y ", t) if p.strip()]

    counts = defaultdict(int)
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

        # "otra/otro" sin producto explícito -> último
        if not product_phrase and any(w in {"otra", "otro", "otras", "otros"} for w in words):
            product_phrase = last_prod

        if not product_phrase:
            # intenta tomar la cola a partir del número encontrado
            if m:
                tail = frag[m.end():].strip()
                product_phrase = _clean_product_phrase(tail.split())
        if not product_phrase:
            continue

        key = product_phrase
        current = counts[key]

        if is_better:
            counts[key] = qty
        elif is_remove:
            if m:
                counts[key] = max(0, current - qty)
            else:
                counts[key] = 0
        else:
            counts[key] = current + qty

        last_prod = key

    products = [{k: v} for k, v in counts.items() if v > 0]
    return {"data": {"products": products}}

# ========================
# Endpoints
# ========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    mode: str | None = Form(None)  # None (texto crudo) | "order_free" (estructura productos)
):
    # guarda a tmp
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        if mode == "order_free":
            structured = parse_order_free(text)
            return JSONResponse({
                **structured,
                "raw_text": text,
                "language": info.language,
                "duration": info.duration
            })

        # por defecto, responde texto crudo
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
