import os
import json
import hashlib

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

def load_prompts() -> dict:
    """Carga de forma dinámica el catálogo de prompts."""
    if not os.path.exists(PROMPTS_FILE):
        raise FileNotFoundError(f"No se encontró el catálogo de prompts en {PROMPTS_FILE}")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_prompt(prompt_id: str) -> tuple[str, str, str]:
    """
    Carga un prompt del registro y calcula su Hash SHA-256 único.
    
    Parámetros:
    - prompt_id: Identificador único del prompt (ej. 'extraccion_rag', 'qa_rag').
    
    Retorna:
    - tuple: (system_prompt, version, prompt_hash_10_chars)
    """
    prompts = load_prompts()
    if prompt_id not in prompts:
        raise KeyError(f"El prompt_id '{prompt_id}' no está registrado en prompts.json")
        
    prompt_data = prompts[prompt_id]
    system_prompt = prompt_data["system_prompt"]
    version = prompt_data.get("version", "1.0.0")
    
    import subprocess
    
    # Calcular Hash SHA-256 del texto del prompt (10 caracteres para legibilidad)
    texto_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:10]
    
    # Intentar obtener el commit de Git
    git_commit = ""
    try:
        # cwd debe ser un directorio dentro del repositorio (usamos __file__)
        cwd = os.path.dirname(os.path.abspath(__file__))
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            cwd=cwd, 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        git_commit = "no-git"
        
    prompt_hash = f"{texto_hash}-{git_commit}"
    
    return system_prompt, version, prompt_hash
