import pandas as pd
import os
import logging
import docx
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def iter_block_items(parent):
    """
    Genera una referencia a cada párrafo y tabla hijos dentro de *parent*,
    en el orden en el que aparecen en el documento.
    """
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("El elemento padre debe ser un Documento o una Celda")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def limpiar_texto_celda(texto: str) -> str:
    """
    Limpia el texto de una celda para que no rompa el formato de la tabla markdown.
    Reemplaza saltos de línea por espacios.
    """
    if not texto:
        return ""
    return texto.replace('\n', ' ').replace('\r', '').strip()

def extraer_texto_y_tablas_word_a_md(ruta_word: str, ruta_salida_md: str) -> None:
    """
    Lee un documento Word (.docx), extrae sus párrafos y tablas manteniendo el orden,
    y lo guarda como un documento Markdown (.md).
    
    Args:
        ruta_word (str): Ruta al archivo Word de entrada (.docx).
        ruta_salida_md (str): Ruta donde se guardará el archivo Markdown de salida (.md).
    """
    try:
        doc = docx.Document(ruta_word)
        
        with open(ruta_salida_md, 'w', encoding='utf-8') as archivo_md:
            nombre_base = os.path.basename(ruta_word)
            archivo_md.write(f"# Documento: {nombre_base}\n\n")
            
            for item in iter_block_items(doc):
                if isinstance(item, Paragraph):
                    texto = item.text.strip()
                    if texto:
                        # Detección básica de encabezados por estilo
                        estilo = item.style.name.lower() if item.style else ""
                        if 'heading 1' in estilo or 'título 1' in estilo:
                            archivo_md.write(f"## {texto}\n\n")
                        elif 'heading 2' in estilo or 'título 2' in estilo:
                            archivo_md.write(f"### {texto}\n\n")
                        elif 'heading 3' in estilo or 'título 3' in estilo:
                            archivo_md.write(f"#### {texto}\n\n")
                        else:
                            archivo_md.write(f"{texto}\n\n")
                            
                elif isinstance(item, Table):
                    datos_tabla = []
                    # Las tablas en Word a veces tienen filas irregulares (celdas combinadas), 
                    # iteramos protegiendo el acceso a las celdas
                    for row in item.rows:
                        fila_datos = [limpiar_texto_celda(cell.text) for cell in row.cells]
                        datos_tabla.append(fila_datos)
                    
                    if datos_tabla and len(datos_tabla) > 1:
                        # Asumimos que la primera fila es el encabezado
                        df = pd.DataFrame(datos_tabla[1:], columns=datos_tabla[0])
                        # Limpiar nulos para evitar errores
                        df = df.fillna("")
                        
                        # Generamos la tabla en formato markdown
                        tabla_markdown = df.to_markdown(index=False)
                        archivo_md.write(tabla_markdown + "\n\n")
                    elif len(datos_tabla) == 1:
                        # Tabla de una sola fila, la formateamos manualmente como tabla markdown
                        fila = " | ".join(datos_tabla[0])
                        separador = " | ".join(["---"] * len(datos_tabla[0]))
                        archivo_md.write(f"| {fila} |\n")
                        archivo_md.write(f"| {separador} |\n\n")
                        
        logger.info(f"Éxito: Documento Word convertido y guardado en '{ruta_salida_md}'")
        
    except FileNotFoundError:
        logger.error(f"El archivo no fue encontrado en la ruta: {ruta_word}")
    except Exception as e:
        logger.exception(f"Error inesperado al procesar el archivo Word: {e}")

# --- Prueba ---
if __name__ == '__main__':
    import sys
    # Extraer el archivo de ayudas IFRS9 como prueba si se ejecuta directamente
    entrada = os.path.join("data", "01_raw", "AYUDAS CNR IFRS9_20250825.docx")
    salida = os.path.join("data", "02_interim", "AYUDAS_CNR_IFRS9.md")
    if os.path.exists(entrada):
        extraer_texto_y_tablas_word_a_md(entrada, salida)
