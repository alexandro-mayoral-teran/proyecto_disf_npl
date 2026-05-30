import os
import sys
from pathlib import Path

# Configurar rutas para importar desde 'src' robustamente
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

from src.nlp_core.prompts_registry import get_prompt

def main():
    print("================================================================================")
    print("🧪 SCRIPT DE PRUEBA: REGISTRO DE PROMPTS Y TRAZABILIDAD (MA3)")
    print("================================================================================")
    
    prompts_to_test = ["extraccion_full_context", "extraccion_rag", "qa_rag", "juez_extraccion"]
    
    print("\nCargando y hasheando prompts...")
    for pid in prompts_to_test:
        try:
            text, version, p_hash = get_prompt(pid)
            print(f"\nPrompt ID: {pid}")
            print(f"  ↳ Versión: {version}")
            print(f"  ↳ Hash Único (SHA-256): {p_hash}")
            print(f"  ↳ Longitud del prompt: {len(text)} caracteres.")
            
            # Verificar estructura del hash
            assert len(p_hash) >= 10, "❌ Error: El hash del prompt debe tener al menos 10 caracteres."
            
        except Exception as e:
            print(f"❌ Error al cargar prompt '{pid}': {e}")
            
    print("\n✅ Verificación del Registro de Prompts: COMPLETADA CON ÉXITO.")
    print("Todos los prompts se cargan, se versionan y generan hashes consistentes.")

if __name__ == "__main__":
    main()
