from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import os
import sys
import yaml

# Forzar UTF-8 en stdout para evitar errores de charmap con emojis en Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Agregar src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.nlp_core.generacion import extraer_rag_cascade, responder_rag_cascade_qa, extraer_full_context, extraer_metadatos_documento
from datetime import datetime

app = FastAPI(title="API DISF - Especialista Digital Regulador")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    tema: Optional[str] = None
    textos_efimeros: Optional[List[str]] = None
    solo_efimero: bool = False
    db_folder: Optional[str] = "chroma_db"
    instrucciones: Optional[str] = None

@app.get("/api/databases")
def get_databases():
    base_dir = os.path.join(os.path.dirname(__file__), '../data/03_output')
    dbs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if item.startswith("chroma_db") and os.path.isdir(os.path.join(base_dir, item)):
                # Generar etiqueta amigable
                label = item.replace("chroma_db", "").replace("_", " ").strip()
                if not label:
                    label = "Actual (chroma_db)"
                else:
                    label = label.title()
                dbs.append({"value": item, "label": label})
    
    # Ordenar asegurando que 'chroma_db' quede de primero
    dbs.sort(key=lambda x: (x["value"] != "chroma_db", x["label"]))
    return {"status": "success", "databases": dbs}

@app.post("/api/extraer_formulario")
def extraer_formulario_endpoint(request: ChatRequest):
    try:
        # Extracción Pydantic con Cascade
        kwargs = {
            "query": request.query,
            "tema": request.tema,
            "textos_efimeros": request.textos_efimeros,
            "solo_efimero": request.solo_efimero,
            "db_folder": request.db_folder
        }
            
        resultado_rag, telemetria = extraer_rag_cascade(**kwargs)
        
        # FastAPI convertirá automáticamente el modelo Pydantic a JSON
        return {
            "status": "success",
            "data": resultado_rag.model_dump(),
            "telemetry": telemetria
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del servidor: {str(e)}")

@app.post("/api/extraer_formulario_full_context")
def extraer_formulario_full_context_endpoint(request: ChatRequest):
    """
    Nuevo endpoint que ignora el RAG y pasa todo el texto concatenado al LLM usando 
    la ventana de contexto larga (128k tokens) para no perder ningún campo.
    """
    try:
        if not request.textos_efimeros or len(request.textos_efimeros) == 0:
            raise ValueError("Debes proporcionar al menos un documento para la extracción de contexto largo.")
            
        # Unir todos los textos efímeros en un gran string
        texto_completo = "\n\n--- SIGUIENTE DOCUMENTO ---\n\n".join(request.textos_efimeros)
        
        # Llamar al motor de extracción Long-Context
        resultado_pydantic, telemetria = extraer_full_context(texto_completo, instrucciones=request.instrucciones)
        data_dict = resultado_pydantic.model_dump()
        
        # Guardar en output local
        import json
        output_dir = os.path.join(os.path.dirname(__file__), '../data/03_output/formularios_extraidos')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"formulario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
        
        return {
            "status": "success",
            "data": data_dict,
            "telemetry": telemetria,
            "saved_to": filename
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/extraer_metadatos")
def extraer_metadatos_endpoint(request: ChatRequest):
    """
    Endpoint que toma el documento temporal cargado y extrae su metadata.
    Guarda el resultado en data/03_output/metadatos_extraidos/.
    """
    import json
    try:
        if not request.textos_efimeros or len(request.textos_efimeros) == 0:
            raise ValueError("Debes proporcionar al menos un documento (Temporal) para la extracción de metadatos.")
            
        texto = request.textos_efimeros[0]
        
        resultado_pydantic, telemetria = extraer_metadatos_documento(texto, instrucciones=request.instrucciones)
        data_dict = resultado_pydantic.model_dump()
        
        # Guardar en output local
        output_dir = os.path.join(os.path.dirname(__file__), '../data/03_output/metadatos_extraidos')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
            
        return {
            "status": "success",
            "data": data_dict,
            "telemetry": telemetria,
            "saved_to": filename
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/consulta_normativa")
def consulta_normativa_endpoint(request: ChatRequest):
    try:
        # RAG Conversacional Cascade (usando los defaults óptimos de generacion.py)
        kwargs = {
            "query": request.query,
            "tema": request.tema,
            "textos_efimeros": request.textos_efimeros,
            "solo_efimero": request.solo_efimero,
            "db_folder": request.db_folder
        }
            
        texto_respuesta, telemetria, contexto = responder_rag_cascade_qa(**kwargs)
        
        return {
            "status": "success",
            "data": texto_respuesta,
            "telemetry": telemetria,
            "context": contexto
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/upload_efimero")
async def upload_efimero_endpoint(file: UploadFile = File(...)):
    import tempfile
    import shutil
    from src.ingesta.ingestor import IngestorDocumentos
    
    try:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        ingestor = IngestorDocumentos(output_dir=temp_dir)
        resultado = ingestor.procesar_archivo(temp_file_path)
        
        if resultado["status"] == "success":
            md_path = os.path.join(temp_dir, resultado["output_file"])
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            return {"status": "success", "markdown": md_text}
        else:
            raise HTTPException(status_code=400, detail=resultado.get("error", "Error procesando el archivo"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/api/evaluaciones")
async def get_evaluaciones():
    try:
        # Rutas a los tres escenarios
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/03_output'))
        
        # 1. Contextualizador (SOTA)
        path_contextualizador = os.path.join(base_path, 'evaluaciones', 'ARENA_RESULTADOS_llm_judge.csv')
        df_contextualizador = pd.read_csv(path_contextualizador)
        
        # 2. Inyector
        path_inyector = os.path.join(base_path, 'evaluaciones_inyector_metadata', 'ARENA_RESULTADOS_llm_judge.csv')
        df_inyector = pd.read_csv(path_inyector)
        
        # 3. Only Chunking
        path_chunking = os.path.join(base_path, 'evaluaciones_only_chunking', 'ARENA_RESULTADOS_llm_judge.csv')
        df_chunking = pd.read_csv(path_chunking)
        
        return {
            "status": "success",
            "scenarios": {
                "only_chunking": df_chunking.to_dict(orient="records"),
                "inyector": df_inyector.to_dict(orient="records"),
                "contextualizador": df_contextualizador.to_dict(orient="records")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer evaluaciones: {str(e)}")

@app.get("/api/temas")
def get_temas():
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/01_raw/manifest.yaml'))
    temas = set()
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = yaml.safe_load(f)
            if manifest_data and 'documentos' in manifest_data:
                for doc in manifest_data['documentos']:
                    if 'tema' in doc:
                        if isinstance(doc['tema'], list):
                            for t in doc['tema']:
                                temas.add(t)
                        else:
                            temas.add(doc['tema'])
        return {"status": "success", "temas": sorted(list(temas))}
    except Exception as e:
        return {"status": "error", "temas": [], "detail": str(e)}

# Montar los archivos estáticos de la app (Frontend Vanilla JS)
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app'))
app.mount("/", StaticFiles(directory=app_path, html=True), name="app")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main_api:app", host="127.0.0.1", port=8000, reload=True)
