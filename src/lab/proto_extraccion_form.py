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

# Definir las clases Pydantic para el formulario
class CampoFormulario(BaseModel):
    nombre_campo: str = Field(description="Nombre corto o mnemónico de la variable (ej. PAGO, ATR_i, SDOIMP).")
    tipo_dato: str = Field(description="Tipo de dato sugerido (ej. Decimal, Entero, Fecha, Booleano).")
    descripcion_funcional: str = Field(description="Descripción de qué significa este campo según la normativa.")
    formula_calculo: Optional[str] = Field(description="Si el campo se calcula mediante una fórmula o condición matemática explícita en el texto, descríbela aquí.")
    obligatorio: bool = Field(description="Indica si el campo es de provisión obligatoria según el texto.")

class FormularioExtraido(BaseModel):
    nombre_formulario: str = Field(description="Nombre sugerido para el formulario o reporte regulatorio.")
    campos: List[CampoFormulario] = Field(description="Lista de campos extraídos de la normativa.")
    frecuencia_reporte: Optional[str] = Field(description="Periodicidad con la que se debe calcular/reportar, si se menciona.")

def extraer_formulario():
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "02_interim" / "markdown" / "CUB_extracto.md"
    
    if not file_path.exists():
        print(f"Error: No se encontró el archivo en {file_path}")
        return
        
    print(f"Leyendo documento normativo: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    print(f"Tamaño del documento cargado en memoria: {len(content)} caracteres.")
    print("\nIniciando extracción estructurada de formularios (Long-Context)...")
    print("Esto podría tomar unos 10-20 segundos dependiendo del tamaño del texto.")
    
    # Usaremos gpt-4o-mini porque soporta 128k tokens, suficiente para este documento
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(FormularioExtraido)
    
    prompt = f"""
    Eres un Analista Regulador Experto de la DISF del Banco de México.
    A continuación se presenta un capítulo normativo completo. 
    Tu objetivo es leerlo íntegramente e identificar todas las variables, métricas, insumos y reglas matemáticas requeridas para calcular las reservas preventivas de la cartera crediticia.
    Con base en estas reglas, propón la estructura de la base de datos (formulario) que las instituciones deberán enviar para cumplir con este marco regulatorio.
    
    --- NORMATIVA ---
    {content}
    """
    
    resultado = structured_llm.invoke(prompt)
    
    print("\n================ FORMULARIO EXTRAÍDO ================")
    print(f"Nombre: {resultado.nombre_formulario}")
    print(f"Frecuencia: {resultado.frecuencia_reporte}\n")
    print("CAMPOS:")
    for i, campo in enumerate(resultado.campos):
        formula = f"\n    Fórmula: {campo.formula_calculo}" if campo.formula_calculo else ""
        print(f"{i+1}. {campo.nombre_campo} ({campo.tipo_dato}) - Obligatorio: {campo.obligatorio}")
        print(f"    Desc: {campo.descripcion_funcional}{formula}")
    print("=====================================================")

if __name__ == "__main__":
    extraer_formulario()
