import os
import sys
from dotenv import load_dotenv

# Asegurar que el directorio raíz del proyecto (directorio padre de 'notebooks') esté en el sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

from src.nlp_core.config_llm import (
    get_llm_client, 
    get_langchain_chat, 
    is_local_llm_enabled,
    get_embeddings,
    is_local_embeddings_enabled
)

def probar_openai():
    print("\n=== Probando OpenAI Nube ===")
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY no encontrada en el .env.")
        return False
    try:
        # Usamos gpt-4o-mini para hacer una prueba rápida
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "di hola en una palabra"}],
            max_tokens=5
        )
        texto = response.choices[0].message.content.strip()
        print(f"✅ Conexión con OpenAI exitosa. Respuesta: '{texto}'")
        return True
    except Exception as e:
        print(f"❌ Error al conectar con OpenAI: {e}")
        return False

def probar_local():
    print("\n=== Probando Ollama / Modelo Local ===")
    local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
    print(f"Configurado en: {local_url}")
    
    try:
        from openai import OpenAI
        client = OpenAI(base_url=local_url, api_key="ollama")
        
        # Intentamos listar modelos disponibles
        try:
            modelos = client.models.list()
            print("Modelos locales disponibles en Ollama:")
            for m in modelos.data:
                print(f" - {m.id}")
        except Exception as err_list:
            print(f"⚠️ No se pudieron listar los modelos locales: {err_list}")
            
        model_name = os.getenv("LLM_MODEL_QA", "llama3.1:8b")
        print(f"Probando chat con el modelo local '{model_name}'...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "di hola en una palabra"}],
            max_tokens=5
        )
        texto = response.choices[0].message.content.strip()
        print(f"✅ Conexión con Ollama exitosa. Respuesta: '{texto}'")
        return True
    except Exception as e:
        print(f"❌ Error al conectar con Ollama: {e}")
        print("Asegúrate de que Ollama esté corriendo y hayas ejecutado 'ollama run llama3.1:8b'")
        return False

def probar_embeddings():
    print("\n=== Probando Embeddings ===")
    try:
        is_local = is_local_embeddings_enabled()
        print(f"Embeddings configurados como: {'LOCAL' if is_local else 'NUBE (OpenAI)'}")
        
        embedder = get_embeddings()
        vector = embedder.embed_query("Hola mundo")
        print(f"✅ Embeddings generados exitosamente. Dimensión del vector: {len(vector)}")
        return True
    except Exception as e:
        print(f"❌ Error al generar embeddings: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    print("Iniciando prueba de conexiones...")
    print("Variables de entorno cargadas.")
    
    # Mostrar configuraciones
    print(f"USE_LOCAL_QA: {os.getenv('USE_LOCAL_QA')}")
    print(f"USE_LOCAL_EXTRACTION: {os.getenv('USE_LOCAL_EXTRACTION')}")
    print(f"USE_LOCAL_EXPANSION: {os.getenv('USE_LOCAL_EXPANSION')}")
    print(f"USE_LOCAL_JUDGE: {os.getenv('USE_LOCAL_JUDGE')}")
    
    probar_openai()
    probar_local()
    probar_embeddings()
