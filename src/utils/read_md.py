from pathlib import Path
import os

def read_markdown_file(markdown_path: Path) -> dict:
    """
    Lee un archivo markdown convertido previamente como diccionario.
    """
    with open(markdown_path, "r", encoding="utf-8") as file:
        text = file.read()

    return {
        "file_name": markdown_path.name,
        "file_stem": markdown_path.stem,
        "file_path": str(markdown_path),
        "text": text,
        "character_count": len(text),
        "word_count": len(text.split())
    }
