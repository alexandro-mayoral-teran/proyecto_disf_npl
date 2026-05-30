import os
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def is_local_llm_enabled(task: str = "qa") -> bool:
    """Verifica si el uso de un LLM local está habilitado para una tarea específica (Enfoque Híbrido)."""
    env_var_name = f"USE_LOCAL_{task.upper()}"
    return os.getenv(env_var_name, "false").lower() == "true"

def get_llm_model_name(task: str = "extraction") -> str:
    """Devuelve el nombre del modelo a usar, ya sea local o de OpenAI, dependiendo de la tarea."""
    if is_local_llm_enabled(task):
        if task == "extraction":
            return os.getenv("LLM_MODEL_EXTRACTION", "llama3.1:8b")
        elif task == "judge":
            return os.getenv("LLM_MODEL_JUDGE", "llama3.1:8b")
        else:
            return os.getenv("LLM_MODEL_QA", "llama3.1:8b")
    else:
        if task == "extraction":
            return os.getenv("LLM_MODEL_EXTRACTION_CLOUD", "gpt-4o")
        elif task == "judge":
            return os.getenv("LLM_MODEL_JUDGE_CLOUD", "gpt-4o")
        else:
            return os.getenv("LLM_MODEL_QA_CLOUD", "gpt-4o-mini")

def get_llm_client(task: str = "extraction") -> OpenAI:
    """Devuelve un cliente base de OpenAI, configurado para apuntar a la nube o a un servidor local (ej. Ollama)."""
    if is_local_llm_enabled(task):
        return OpenAI(
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            api_key="ollama" # dummy api key para modelos locales
        )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")
        return OpenAI() # Usa OPENAI_API_KEY nativamente

def get_langchain_chat(task: str = "qa", temperature: float = 0.0) -> ChatOpenAI:
    """Devuelve un ChatOpenAI (LangChain) apuntando al proveedor configurado."""
    model_name = get_llm_model_name(task)
    
    if is_local_llm_enabled(task):
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            api_key="ollama"
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

def is_local_embeddings_enabled() -> bool:
    """Verifica si el uso de embeddings locales está habilitado."""
    return os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

def get_embeddings() -> OpenAIEmbeddings:
    """Devuelve la clase de Embeddings para LangChain (OpenAI o Local compatible)."""
    if is_local_embeddings_enabled():
        # Utiliza el endpoint compatible de OpenAI de Ollama para embeddings
        return OpenAIEmbeddings(
            model=os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            api_key="ollama"
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key
        )
