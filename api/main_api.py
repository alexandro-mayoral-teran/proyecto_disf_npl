from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.nlp_core.agente import extraer_rag_simple, responder_rag_qa

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
    top_k: int = 4
    base_retriever: str = "embeddings"
    query_expansion: str = "none"
    post_processing: str = "none"

@app.post("/api/extraer_formulario")
async def extraer_formulario_endpoint(request: ChatRequest):
    try:
        # Extracción Pydantic
        resultado_rag, telemetria = extraer_rag_simple(request.query, k=request.top_k)
        
        # FastAPI convertirá automáticamente el modelo Pydantic a JSON
        return {
            "status": "success",
            "data": resultado_rag.model_dump(),
            "telemetry": telemetria
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/consulta_normativa")
async def consulta_normativa_endpoint(request: ChatRequest):
    try:
        # RAG Conversacional puro con configuración dinámica de Pipeline
        texto_respuesta, telemetria, contexto = responder_rag_qa(
            request.query, 
            k=request.top_k,
            base_retriever=request.base_retriever,
            query_expansion=request.query_expansion,
            post_processing=request.post_processing
        )
        
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

# Montar los archivos estáticos de la app (Frontend Vanilla JS)
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app'))
app.mount("/", StaticFiles(directory=app_path, html=True), name="app")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main_api:app", host="127.0.0.1", port=8000, reload=True)
