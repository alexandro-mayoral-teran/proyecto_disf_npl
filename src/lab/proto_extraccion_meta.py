import os
import sys
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# Forzar UTF-8
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cargar variables de entorno
load_dotenv()

# Añadir src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from langchain_openai import ChatOpenAI

class MetadataDocumento(BaseModel):
    tema_principal: str = Field(description="El tema principal del documento (ej. Crédito, Riesgos, Liquidez).")
    subtemas: List[str] = Field(description="Lista de subtemas secundarios.")
    nivel_confidencialidad: str = Field(description="Nivel de privacidad: 'Publico', 'Interno', 'Restringido'. Si no se menciona, asume 'Publico'.")
    audiencia: str = Field(description="A quién va dirigido el documento (ej. Bancos, SOFOMES, Público General).")
    frecuencia_reporte: Optional[str] = Field(description="Si el documento menciona una periodicidad de reporte (ej. Mensual, Bimestral), indícalo aquí.")
    descripcion_corta: str = Field(description="Un resumen muy breve (1 oración) de lo que trata el documento.")

def extraer_metadatos():
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "02_interim" / "markdown" / "AYUDAS_CNR_IFRS9.md"
    
    if not file_path.exists():
        print(f"Error: No se encontró el archivo en {file_path}")
        return
        
    print(f"Leyendo documento: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Usaremos solo los primeros 4000 caracteres para extraer metadatos (ahorro de tokens y velocidad)
    texto_para_analizar = content[:4000]
    
    print("\nIniciando extracción estructurada de metadatos (Gobernanza)...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(MetadataDocumento)
    
    prompt = f"""
    Eres un bibliotecario y experto en gobierno de datos del Banco de México (DISF).
    Analiza el siguiente fragmento inicial de un documento normativo y extrae la metadata solicitada.
    
    --- DOCUMENTO ---
    {texto_para_analizar}
    """
    
    resultado = structured_llm.invoke(prompt)
    
    print("\n================ METADATOS EXTRAÍDOS ================")
    print(json.dumps(resultado.model_dump(), indent=2, ensure_ascii=False))
    print("=====================================================")

if __name__ == "__main__":
    extraer_metadatos()
