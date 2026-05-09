import os
import sys

# Agregar la raíz al sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.limpieza_texto import procesar_documento
from src.nlp_core.agente import extraer_requerimiento_openai

def probar_con_archivo_real():
    ruta_md = "data/02_interim/CUB_extracto.md"
    if not os.path.exists(ruta_md):
        print(f"No se encontró el archivo: {ruta_md}")
        return
        
    print(f"📄 Leyendo documento real: {ruta_md}...")
    with open(ruta_md, "r", encoding="utf-8") as f:
        texto_crudo = f.read()
        
    print("🧹 Limpiando ruido del documento...")
    texto_limpio = procesar_documento(texto_crudo, origen="CNBV")
    
    print(f"📏 Texto listo. Longitud: {len(texto_limpio)} caracteres.")
    print("🚀 Enviando a OpenAI (gpt-4o-mini). Esto puede tardar un poco y consumir más tokens...")
    
    try:
        resultado = extraer_requerimiento_openai(texto_limpio)
        print("\n✅ ¡Extracción exitosa!")
        
        # Guardar en un JSON para poder revisarlo con calma
        ruta_salida = "data/03_output/resultado_ayudas_cnr.json"
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        
        with open(ruta_salida, "w", encoding="utf-8") as f:
            # model_dump_json serializa correctamente el objeto de pydantic
            f.write(resultado.model_dump_json(indent=4))
            
        print(f"\n📁 Resultado completo guardado en: {ruta_salida}")
        print(f"📝 Formulario detectado: {resultado.nombre_formulario}")
        print(f"📊 Campos extraídos: {len(resultado.campos_formulario)}")
        print(f"🗂️ Catálogos detectados: {len(resultado.catalogos_identificados)}")
        print(f"⚠️ Ambigüedades/Observaciones: {len(resultado.ambiguedades_detectadas) if resultado.ambiguedades_detectadas else 0}")
        
    except Exception as e:
        print(f"❌ Error durante la extracción: {e}")

if __name__ == "__main__":
    probar_con_archivo_real()
