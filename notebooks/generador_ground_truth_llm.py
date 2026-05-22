import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Cargar variables de entorno (Asegúrate de tener OPENAI_API_KEY en tu .env)
load_dotenv()

# ==============================================================================
# 1. Definición del Esquema (Pydantic)
# ==============================================================================
class DocumentoEsperado(BaseModel):
    archivo: str = Field(description="Nombre del archivo Markdown de origen")
    jerarquia_esperada: str = Field(description="Jerarquía de la sección donde se encuentra la respuesta (ej. Artículo 91 -> II. Tipos)")
    texto_clave_esperado: str = Field(description="Un fragmento literal EXACTO de 15 a 30 palabras extraído del texto que responde a la pregunta. Debe ser copiado y pegado exactamente.")
    justificacion: str = Field(description="Por qué este documento responde a la pregunta.")

class PreguntaSintetica(BaseModel):
    pregunta: str = Field(description="Pregunta analítica y compleja sobre la regulación. Debe sonar como algo que preguntaría un analista financiero.")
    dificultad: str = Field(description="Baja, Media o Alta")
    documentos_esperados: list[DocumentoEsperado]

class LotePreguntas(BaseModel):
    preguntas: list[PreguntaSintetica]

# ==============================================================================
# 2. Configuración del LLM y Prompt
# ==============================================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
llm_estructurado = llm.with_structured_output(LotePreguntas)

system_prompt = """
Eres un Analista Experto del Banco de México (DISF).
Tu tarea es leer un fragmento de una regulación financiera y generar {num_preguntas} preguntas difíciles y representativas que un auditor o analista intentaría responder usando este texto.

REGLA DE ORO PARA EL TEXTO CLAVE:
Para hacer que la evaluación sea agnóstica a la técnica de chunking, debes extraer un `texto_clave_esperado`. 
Este DEBE ser una cita textual, literal y exacta de una oración clave dentro del fragmento que responde a la pregunta.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Archivo origen: {nombre_archivo}\nJerarquía: {jerarquia}\n\nTexto:\n{texto_fragmento}")
])

chain = prompt | llm_estructurado

# ==============================================================================
# 3. Función Principal de Generación
# ==============================================================================
def generar_preguntas_desde_markdown(ruta_md: str, archivo_salida_json: str, preguntas_por_fragmento: int = 1):
    print(f"Leyendo documento: {ruta_md}")
    with open(ruta_md, 'r', encoding='utf-8') as f:
        contenido = f.read()

    # Usamos el splitter por headers para dividir el documento en secciones semánticas
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragmentos = markdown_splitter.split_text(contenido)
    
    nombre_archivo = os.path.basename(ruta_md)
    preguntas_totales = []
    
    # Procesamos solo algunos fragmentos largos para no gastar demasiados tokens de golpe (ej. los primeros 5 largos)
    fragmentos_utiles = [f for f in fragmentos if len(f.page_content) > 500][:5]
    
    print(f"Generando preguntas para {len(fragmentos_utiles)} fragmentos extensos encontrados...")
    
    for i, fragmento in enumerate(fragmentos_utiles):
        print(f"Procesando fragmento {i+1}/{len(fragmentos_utiles)}...")
        
        # Construir el string de jerarquía a partir de la metadata
        jerarquia = " -> ".join([val for key, val in fragmento.metadata.items()])
        if not jerarquia:
            jerarquia = "Sección General"
            
        try:
            resultado = chain.invoke({
                "num_preguntas": preguntas_por_fragmento,
                "nombre_archivo": nombre_archivo,
                "jerarquia": jerarquia,
                "texto_fragmento": fragmento.page_content
            })
            preguntas_totales.extend([p.model_dump() for p in resultado.preguntas])
        except Exception as e:
            print(f"Error procesando fragmento {i+1}: {e}")
            
    # Guardar o actualizar el JSON
    salida_final = []
    if os.path.exists(archivo_salida_json):
        with open(archivo_salida_json, 'r', encoding='utf-8') as f:
            try:
                salida_final = json.load(f)
            except:
                pass
                
    # Asignar IDs a las nuevas
    start_id = len(salida_final) + 1
    for idx, p in enumerate(preguntas_totales):
        p['query_id'] = f"Q{start_id + idx:02d}"
        salida_final.append(p)
        
    with open(archivo_salida_json, 'w', encoding='utf-8') as f:
        json.dump(salida_final, f, indent=2, ensure_ascii=False)
        
    print(f"\n¡Éxito! Se añadieron {len(preguntas_totales)} preguntas sintéticas a {archivo_salida_json}")

if __name__ == "__main__":
    # Ruta de ejemplo. Ajusta según sea necesario.
    ruta_markdown_prueba = "../data/02_interim/markdown/CUB_extracto.md"
    ruta_salida = "../data/evaluacion_dataset.json"
    
    # Asegúrate de ejecutar el script desde el directorio notebooks/
    generar_preguntas_desde_markdown(ruta_markdown_prueba, ruta_salida, preguntas_por_fragmento=3)
