import os
import sys
import time
import requests
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Dependencias para Word
import docx
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestorDocumentos:
    """
    Clase para la ingesta de documentos regulatorios.
    Detecta automáticamente el tipo de archivo y lo convierte a Markdown (.md)
    para alimentar el flujo de vectorización (RAG).
    Soporta: PDF (vía BlazeDocs), Excel (.xlsx) y Word (.docx).
    """
    
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        load_dotenv()
        self.blazedocs_api_key = os.getenv("BLAZEDOCS_API_KEY")
        self.blazedocs_url = "https://blazedocs.io/api/v1/convert"

    def procesar_archivo(self, input_path: str | Path, overwrite: bool = False) -> dict:
        """
        Procesa un solo archivo (PDF, Word, Excel) y lo convierte a Markdown.
        Retorna un diccionario con el log de la conversión.
        """
        input_path = Path(input_path)
        
        log_entry = {
            "file_name": input_path.name,
            "output_file": f"{input_path.stem}.md",
            "status": "error",
            "page_count": None,
            "token_count": None,
            "processing_time_ms": None,
            "error": None
        }

        if not input_path.exists():
            log_entry["error"] = "El archivo no existe"
            logger.error(f"El archivo no existe: {input_path}")
            return log_entry

        output_path = self.output_dir / f"{input_path.stem}.md"

        if output_path.exists() and not overwrite:
            logger.info(f"Saltando archivo (ya existe): {output_path.name}")
            log_entry["status"] = "skipped_already_exists"
            return log_entry

        ext = input_path.suffix.lower()
        logger.info(f"Procesando '{input_path.name}'...")
        
        try:
            start_time = time.time()
            data_meta = {}
            
            if ext == '.pdf':
                data_meta = self._procesar_pdf(input_path, output_path)
            elif ext in ['.xlsx', '.xls']:
                data_meta = self._procesar_excel(input_path, output_path)
            elif ext == '.docx':
                data_meta = self._procesar_word(input_path, output_path)
            else:
                log_entry["error"] = f"Extensión no soportada: {ext}"
                logger.warning(log_entry["error"])
                return log_entry
                
            end_time = time.time()
            logger.info(f"Éxito: {output_path.name} generado correctamente.")
            
            log_entry["status"] = "success"
            log_entry["page_count"] = data_meta.get("page_count")
            log_entry["token_count"] = data_meta.get("token_count")
            # Usa el tiempo del API si existe, sino lo calcula localmente
            log_entry["processing_time_ms"] = data_meta.get("processing_time_ms", int((end_time - start_time) * 1000))
            
            return log_entry
            
        except Exception as e:
            logger.exception(f"Error procesando {input_path.name}: {str(e)}")
            log_entry["error"] = str(e)
            return log_entry

    def procesar_directorio(self, input_dir: str | Path, overwrite: bool = False, pause_seconds: float = 1.0) -> pd.DataFrame:
        """
        Procesa todos los archivos soportados encontrados en un directorio.
        Retorna un DataFrame de Pandas con el log de todas las conversiones, 
        manteniendo compatibilidad con implementaciones previas.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            logger.error(f"El directorio no existe: {input_dir}")
            return pd.DataFrame()

        archivos = []
        archivos.extend(input_dir.glob("*.pdf"))
        archivos.extend(input_dir.glob("*.xlsx"))
        archivos.extend(input_dir.glob("*.docx"))

        conversion_log = []
        for archivo in sorted(archivos):
            res_dict = self.procesar_archivo(archivo, overwrite=overwrite)
            conversion_log.append(res_dict)
            
            # Pausa solo para PDFs que usaron la API y fueron procesados
            if archivo.suffix.lower() == '.pdf' and res_dict.get("status") == "success":
                time.sleep(pause_seconds)
                
        return pd.DataFrame(conversion_log)

    @staticmethod
    def leer_excel_estructurado(input_path: str | Path, tipo: str) -> dict:
        """
        Lee cada pestaña de un excel y devuelve un diccionario de dataframes.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        xls = pd.ExcelFile(input_path)
        tablas = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(input_path, sheet_name=sheet)
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            df = df.dropna(how="all").reset_index(drop=True)
            df.attrs["archivo"] = input_path.name
            df.attrs["tipo"] = tipo
            df.attrs["sheet"] = sheet
            tablas[sheet] = df
        return tablas

    # ==========================================
    # LÓGICA INTERNA (Privada) POR EXTENSIÓN
    # ==========================================

    def _procesar_pdf(self, input_path: Path, output_path: Path) -> dict:
        if not self.blazedocs_api_key:
            raise ValueError("No se encontró BLAZEDOCS_API_KEY en el entorno.")

        headers = {"Authorization": f"Bearer {self.blazedocs_api_key}"}
        with open(input_path, "rb") as file:
            files = {"file": (input_path.name, file, "application/pdf")}
            response = requests.post(self.blazedocs_url, headers=headers, files=files)

        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
            raise ValueError(f"BlazeDocs falló: {result.get('error', 'Error desconocido')}")

        data = result["data"]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(data["markdown"])
            
        return data  # Devuelve metadata (page_count, token_count, processing_time_ms)

    def _procesar_excel(self, input_path: Path, output_path: Path) -> dict:
        archivo_xls = pd.ExcelFile(input_path)
        hojas = archivo_xls.sheet_names
        
        with open(output_path, 'w', encoding='utf-8') as archivo_md:
            archivo_md.write(f"# Documento: {input_path.stem}\n\n")
            
            for hoja in hojas:
                df = pd.read_excel(input_path, sheet_name=hoja)
                df = df.fillna("") 
                archivo_md.write(f"## Tabla / Pestaña: {hoja}\n\n")
                archivo_md.write(df.to_markdown(index=False) + "\n\n")
                
        return {"page_count": len(hojas)}  # Usamos las hojas como aproximación a páginas

    def _procesar_word(self, input_path: Path, output_path: Path) -> dict:
        doc = docx.Document(input_path)
        
        def iter_block_items(parent):
            if isinstance(parent, Document):
                parent_elm = parent.element.body
            elif isinstance(parent, _Cell):
                parent_elm = parent._tc
            else:
                return

            for child in parent_elm.iterchildren():
                if isinstance(child, CT_P):
                    yield Paragraph(child, parent)
                elif isinstance(child, CT_Tbl):
                    yield Table(child, parent)

        def limpiar_texto_celda(texto: str) -> str:
            return texto.replace('\n', ' ').replace('\r', '').strip() if texto else ""

        with open(output_path, 'w', encoding='utf-8') as archivo_md:
            archivo_md.write(f"# Documento: {input_path.stem}\n\n")
            
            for item in iter_block_items(doc):
                if isinstance(item, Paragraph):
                    texto = item.text.strip()
                    if texto:
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
                    for row in item.rows:
                        datos_tabla.append([limpiar_texto_celda(cell.text) for cell in row.cells])
                    
                    if datos_tabla and len(datos_tabla) > 1:
                        df = pd.DataFrame(datos_tabla[1:], columns=datos_tabla[0]).fillna("")
                        archivo_md.write(df.to_markdown(index=False) + "\n\n")
                    elif len(datos_tabla) == 1:
                        fila = " | ".join(datos_tabla[0])
                        separador = " | ".join(["---"] * len(datos_tabla[0]))
                        archivo_md.write(f"| {fila} |\n| {separador} |\n\n")
                        
        return {}  # Word no da metadata de tokens o páginas fácilmente
