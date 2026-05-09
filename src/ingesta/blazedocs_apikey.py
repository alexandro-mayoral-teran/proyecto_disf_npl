import dotenv
import os

dotenv.load_dotenv(dotenv_path=dotenv_path)
def get_blazedocs_api_key() -> str:
    """
    Se toma la API key de BlazeDocs desde una variable de ambiente
    """
    api_key = os.getenv("BLAZEDOCS_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró la API key")

    return api_key
