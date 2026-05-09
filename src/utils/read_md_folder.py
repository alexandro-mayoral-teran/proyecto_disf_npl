from Pathlib import Path
import os

def read_markdown_folder(markdown_dir: Path) -> pd.DataFrame:
    """
    Lee todos los archivos markdown y los almacena en un DataFrame.
    """
    markdown_dir = Path(markdown_dir)

    markdown_files = sorted(markdown_dir.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(f"No Markdown files were found in: {markdown_dir}")

    documents = []

    for markdown_path in markdown_files:
        documents.append(read_markdown_file(markdown_path))

    return pd.DataFrame(documents)
