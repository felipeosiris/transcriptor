import os
import tempfile
import json
import re
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

# ====== modelos Pydantic ======

class TextRequest(BaseModel):
    text: str

# ====== endpoints ======

@app.get("/health")
def health():
    """
    Health check endpoint para Railway.
    CRÍTICO: Debe responder 200 OK siempre y rápido (<100ms).
    Railway usa este endpoint para verificar que la app está viva.
    No debe tener lógica pesada ni dependencias del modelo.
    """
    # Siempre devolver 200 OK con status "ok"
    # Railway solo necesita saber que el servidor responde
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "Transcriptor API",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/transcribe",
            "/resume",
            "/order-summary-ia"
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

@app.post("/resume")
async def resume(
    file: UploadFile = File(...),
    language: str | None = Form(None)
):
    """
    Endpoint para transcribir audio y resumir el texto en una sola línea.
    Recibe un archivo de audio, lo transcribe y devuelve un resumen usando el modelo Pegasus de Hugging Face.
    """
    # Verificar que el modelo Whisper esté cargado
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )
    
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_API_URL = "https://router.huggingface.co/hf-inference/models/google/pegasus-xsum"
    
    tmp_path = None
    try:
        # Paso 1: Leer el archivo de audio
        file_content = await file.read()
        if not file_content:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        
        # Paso 2: Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # Paso 3: Transcribir el audio usando Whisper
        segments, info = m.transcribe(tmp_path, language=language, vad_filter=True)
        transcribed_text = "".join(seg.text for seg in segments).strip()
        
        if not transcribed_text:
            return JSONResponse(
                {"error": "No se pudo transcribir el audio o el audio está vacío"},
                status_code=400
            )
        
        # Paso 4: Preparar el texto con el prompt para el resumen
        prompt_text = (
            f"Regresa lo que recibes pero en json"
            f'Texto transcrito: "{transcribed_text}"'
        )
        
        # Paso 5: Realizar la petición a la API de Hugging Face para resumir
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                HF_API_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={"inputs": prompt_text}
            )
            
            # Verificar si la respuesta es exitosa
            if response.status_code != 200:
                error_detail = response.text if hasattr(response, 'text') else "Unknown error"
                return JSONResponse(
                    {"error": f"Error en la API de Hugging Face: {response.status_code} - {error_detail}"},
                    status_code=500
                )
            
            result = response.json()
            
            # Verificar si hay errores en la respuesta
            if isinstance(result, dict) and "error" in result:
                return JSONResponse(
                    {"error": result.get("error", "Error desconocido de la API")},
                    status_code=500
                )
            
            # Extraer el resumen de la respuesta
            # La API de Hugging Face devuelve normalmente: [{"summary_text": "..."}]
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    summary = result[0].get("summary_text", result[0].get("generated_text", ""))
                else:
                    summary = str(result[0])
            elif isinstance(result, dict):
                summary = result.get("summary_text", result.get("generated_text", ""))
            else:
                summary = str(result)
            
            if not summary:
                return JSONResponse(
                    {"error": "No se pudo generar el resumen"},
                    status_code=500
                )
            
            return JSONResponse({
                "summary": summary.strip(),
                "transcribed_text": transcribed_text,
                "language": info.language,
                "duration": info.duration
            })
            
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": "Timeout al conectar con la API de Hugging Face"},
            status_code=500
        )
    except httpx.RequestError as e:
        return JSONResponse(
            {"error": f"Error de conexión: {str(e)}"},
            status_code=500
        )
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

@app.post("/order-summary-ia")
async def order_summary_ia(
    file: UploadFile = File(...),
    language: str | None = Form(None)
):
    """
    Endpoint para transcribir audio y extraer productos y cantidades usando Groq AI.
    Recibe un archivo de audio, lo transcribe y devuelve un JSON con la lista de productos y cantidades.
    """
    # Verificar que el modelo Whisper esté cargado
    m = get_model()
    if m is None:
        return JSONResponse(
            {"error": f"model_load_failed: {load_error or 'unknown'}"},
            status_code=500
        )
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL_NAME = "llama-3.1-8b-instant"  # Modelo estable y disponible en Groq
    
    # Verificar que la clave API esté configurada
    if not GROQ_API_KEY:
        return JSONResponse(
            {"error": "GROQ_API_KEY no está configurada"},
            status_code=500
        )
    
    tmp_path = None
    try:
        # Paso 1: Leer el archivo de audio
        file_content = await file.read()
        if not file_content:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        
        # Paso 2: Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # Paso 3: Transcribir el audio usando Whisper
        segments, info = m.transcribe(tmp_path, language=language, vad_filter=True)
        transcribed_text = "".join(seg.text for seg in segments).strip()
        
        if not transcribed_text:
            return JSONResponse(
                {"error": "No se pudo transcribir el audio o el audio está vacío"},
                status_code=400
            )
        
        # Paso 4: Preparar el prompt para Groq
        prompt = (
            "A partir del siguiente texto transcrito, extrae únicamente la lista FINAL de productos y cantidades pedidos, considerando cualquier corrección mencionada al final. "
            "\n\n"
            "Tu respuesta debe ser exactamente un JSON válido en este formato (sin explicaciones ni texto adicional, solo el JSON): "
            "\n"
            '{"productos": [{"nombre_producto": cantidad}, {"nombre_producto": cantidad}, ...]} '
            "\n\n"
            "Ejemplo: "
            '{"productos": [{"telera": 2}, {"concha": 2}, {"mantecada": 4}]} '
            "\n\n"
            "Asegúrate de que la cantidad sea número entero. "
            "\n\n"
            "Asegúrate de hacer un barrido final para ver si hubo correcciones de cantidades sobre un producto ya mencionado. por ejemplo si dice 'mejor 2 de algo' debe reemplazar lo que ya habia puesto "
            "\n\n"
            "Asegúrate de no agregar el mismo producto, no suma, solo reemplaza la cantidad. "
            "\n\n"
            f"Texto transcrito: {transcribed_text}"
        )
        
        # Realizar la petición a la API de Groq
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Verificar si la respuesta es exitosa
            if response.status_code != 200:
                error_detail = response.text if hasattr(response, 'text') else "Unknown error"
                return JSONResponse(
                    {"error": f"Error en la API de Groq: {response.status_code} - {error_detail}"},
                    status_code=500
                )
            
            result = response.json()
            
            # Extraer el contenido de la respuesta
            if "choices" not in result or len(result["choices"]) == 0:
                return JSONResponse(
                    {"error": "No se pudo obtener respuesta de la API"},
                    status_code=500
                )
            
            content = result["choices"][0].get("message", {}).get("content", "")
            
            if not content:
                return JSONResponse(
                    {"error": "Respuesta vacía de la API"},
                    status_code=500
                )
            
            # Intentar parsear el JSON
            try:
                # Limpiar el contenido: eliminar markdown code blocks si existen
                content_cleaned = content.strip()
                if content_cleaned.startswith("```"):
                    # Eliminar markdown code blocks
                    content_cleaned = re.sub(r'^```(?:json)?\s*', '', content_cleaned)
                    content_cleaned = re.sub(r'\s*```$', '', content_cleaned)
                
                # Intentar parsear el JSON
                parsed_json = json.loads(content_cleaned)
                
                # Validar que tenga la estructura esperada
                if "productos" not in parsed_json:
                    return JSONResponse(
                        {"error": "invalid JSON", "raw": content},
                        status_code=400
                    )
                
                # Agregar información adicional a la respuesta
                result_response = parsed_json.copy()
                result_response["transcribed_text"] = transcribed_text
                result_response["language"] = info.language
                result_response["duration"] = info.duration
                
                return JSONResponse(result_response)
                
            except json.JSONDecodeError as e:
                # Si el JSON es inválido, intentar repararlo
                try:
                    # Intentar encontrar un bloque JSON válido en el texto
                    # Buscar desde la primera llave hasta la última
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_candidate = content[start_idx:end_idx + 1]
                        # Intentar reemplazar comillas simples por dobles si es necesario
                        json_candidate = json_candidate.replace("'", '"')
                        repaired_json = json.loads(json_candidate)
                        if "productos" in repaired_json:
                            # Agregar información adicional a la respuesta
                            result_response = repaired_json.copy()
                            result_response["transcribed_text"] = transcribed_text
                            result_response["language"] = info.language
                            result_response["duration"] = info.duration
                            return JSONResponse(result_response)
                except:
                    pass
                
                # Si no se pudo reparar, devolver error con el contenido raw
                return JSONResponse(
                    {"error": "invalid JSON", "raw": content},
                    status_code=400
                )
            
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": "Timeout al conectar con la API de Groq"},
            status_code=500
        )
    except httpx.RequestError as e:
        return JSONResponse(
            {"error": f"Error de conexión: {str(e)}"},
            status_code=500
        )
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