import os
import sys
# Agregar el directorio raíz del proyecto al PYTHONPATH para que encuentre 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from openai import OpenAI
from dotenv import load_dotenv

# Importamos nuestro esquema Pydantic
from src.nlp_core.schemas import RequerimientoInformacion

# Cargar variables de entorno desde el archivo .env
load_dotenv()

def extraer_requerimiento_openai(texto_normativo: str) -> RequerimientoInformacion:
    """
    Toma el texto limpio de un documento normativo y utiliza OpenAI 
    para extraer la estructura tabular y los catálogos en un JSON estructurado.
    """
    # Validar que la API Key exista
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno. Asegúrate de configurar tu archivo .env.")
        
    client = OpenAI()
    
    prompt_sistema = (
        "Eres un Especialista Digital Regulador de la DISF en el Banco de México, y actúas como un auditor de datos experto. "
        "Tu tarea es analizar el texto normativo y extraer EXHAUSTIVAMENTE la estructura tabular de los formularios requeridos.\n\n"
        "REGLAS ESTRICTAS DE EXTRACCIÓN:\n"
        "1. EXHAUSTIVIDAD: Extrae TODOS los conceptos matemáticos, variables de reporte y parámetros (ej. Probabilidad de Incumplimiento PI, Reservas, Severidad de la Pérdida SP, etc.). No omitas NINGUNA variable clave mencionada en el texto.\n"
        "2. FÓRMULAS: Si una variable se calcula mediante una fórmula o ecuación, OBLIGATORIAMENTE regístrala en el campo 'formula_calculo'.\n"
        "3. CATÁLOGOS: Si detectas que una variable solo puede tomar valores específicos de una lista (ej. Etapas de riesgo 1, 2, 3), CREA el catálogo. Debes OBLIGATORIAMENTE establecer 'es_catalogo=true' en ese campo y poner el nombre exacto en 'nombre_catalogo_vinculado'.\n"
        "4. VALIDACIONES: Genera reglas de negocio lógicas y analíticas en 'validaciones_sugeridas' (ej. 'PI debe estar entre 0 y 1', 'El Monto no puede superar al límite').\n"
        "5. AMBIGÜEDADES: Si la normativa menciona una fórmula pero no define claramente sus variables, o encuentras huecos lógicos, regístralo SIEMPRE en 'ambiguedades_detectadas'."
    )
    
    # Utilizamos Structured Outputs de OpenAI (disponible en pydantic >= 2.0 y openai >= 1.40)
    # garantizando que la salida cumpla perfectamente con nuestro esquema.
    respuesta = client.beta.chat.completions.parse(
        model="gpt-4o", # Subimos a gpt-4o para asegurar la extracción perfecta de fórmulas complejas
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Texto normativo a analizar:\n\n{texto_normativo}"}
        ],
        response_format=RequerimientoInformacion,
        temperature=0.1 # Temperatura baja porque queremos extracción precisa, no creatividad
    )
    
    # La API ya nos devuelve el objeto Pydantic instanciado y validado
    return respuesta.choices[0].message.parsed

# --- Prueba rápida ---
if __name__ == "__main__":
    # Texto ficticio de prueba (muy sencillo para no gastar muchos tokens)
    texto_prueba = """
    Artículo 1. Las instituciones de crédito deberán enviar mensualmente un reporte de sus créditos comerciales a la DISF.
    Dicho reporte, que denominaremos "Formulario de Créditos Comerciales Mensual", deberá contener:
    1. Identificador del Crédito: Debe ser Alfanumérico con máximo 15 caracteres.
    2. Moneda del crédito: Es obligatorio enviar la clave de moneda. Las opciones válidas son 'MXN' para Pesos Mexicanos y 'USD' para Dólares Estadounidenses.
    3. Tasa de Interés: Debe ser un valor Numérico sin límite. Ojo, esta tasa no puede ser negativa en ningún caso.
    """
    
    print("Iniciando prueba con OpenAI...")
    try:
        resultado = extraer_requerimiento_openai(texto_prueba)
        print("\n¡Extracción exitosa!\n")
        print(f"📊 Formulario propuesto: {resultado.nombre_formulario}")
        print("\n📝 Campos identificados:")
        for campo in resultado.campos_formulario:
            print(f" - {campo.nombre_campo} ({campo.tipo_dato}): {campo.descripcion_funcional}")
            if campo.longitud:
                print(f"   [Longitud: {campo.longitud}]")
            if campo.validaciones_sugeridas:
                print(f"   [Validaciones: {campo.validaciones_sugeridas}]")
            if campo.es_catalogo:
                print(f"   [Vinculado al catálogo: {campo.nombre_catalogo_vinculado}]")
                
        print("\n🗂️ Catálogos identificados:")
        for catalogo in resultado.catalogos_identificados:
            valores = [f"{v.clave} ({v.descripcion})" for v in catalogo.valores]
            print(f" - {catalogo.nombre_catalogo}: {', '.join(valores)}")
            
        if resultado.ambiguedades_detectadas:
            print("\n⚠️ Ambigüedades detectadas:")
            for amb in resultado.ambiguedades_detectadas:
                print(f" - {amb}")
                
    except ValueError as ve:
        print(f"⚠️ Atención: {ve}")
    except Exception as e:
        print(f"❌ Error durante la ejecución con OpenAI: {e}")
