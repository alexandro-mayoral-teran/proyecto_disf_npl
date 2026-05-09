import re

def limpieza_universal(texto: str) -> str:
    """
    Limpieza genérica aplicable a cualquier documento normativo extraído.
    Elimina caracteres residuales, dobles espacios y saltos de línea excesivos.
    """
    # Eliminar símbolos residuales extraños de parsers OCR o PDF
    texto = re.sub(r'\^\{\}|\[\]', '', texto)
    texto = re.sub(r'\x0c', ' ', texto) # Saltos de página ocultos

    # Limpiar los saltos de línea excesivos (más de 2 seguidos se vuelven 2)
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    # Quitar espacios múltiples horizontales
    texto = re.sub(r'[ \t]+', ' ', texto)
    return texto.strip()

def limpiar_ruido_cnbv(texto_md: str) -> str:
    """
    Limpia el ruido específico de documentos de la CNBV (ej. Circular Única de Bancos).
    """
    texto = texto_md

    # Quitar las marcas de control de la CNBV al inicio de línea (ej: "(274) ", "(214) ")
    texto = re.sub(r'^\(\d+\)\s*', '', texto, flags=re.MULTILINE)

    # Quitar el pie de página recurrente (dirección y teléfonos de CNBV)
    texto = re.sub(r'Insurgentes Sur 1971.*?www\.gob\.mx/cnbv', '', texto, flags=re.DOTALL)

    # Quitar encabezados institucionales repetitivos de CNBV / SHCP
    patrones = [
        r'^Hacienda Secretaria de Hacienda y Credito Publico$',
        r'^Hacienda Secretaría de Hacienda y Crédito Público$',
        r'^CNBV$',
        r'^CONSEJO NACIONAL.*$'
    ]

    for patron in patrones:
        texto = re.sub(patron, '', texto, flags=re.MULTILINE)

    return texto.strip()

def limpiar_ruido_banxico(texto: str) -> str:
    """
    Limpia ruido específico de documentos del Banco de México (Circulares, disposiciones, etc.)
    Mantiene la estructura normativa (artículos, títulos), eliminando contenido no informativo.
    """
    texto_limpio = texto

    # 1 Eliminar encabezado institucional repetido
    texto_limpio = re.sub(
        r'BANCO\s+DE\s+M[ÉE]XICO.*?\n',
        '',
        texto_limpio,
        flags=re.IGNORECASE)

    # Eliminar bloque de "TEXTO COMPILADO..."
    # (muy largo, no aporta valor semántico)
    texto_limpio = re.sub(
        r'TEXTO\s+COMPILADO.*?respectivamente\.',
        '',
        texto_limpio,
        flags=re.IGNORECASE | re.DOTALL)

    # Eliminar notas de modificación
    texto_limpio = re.sub(
        r'\(Modificado.*?\)',
        '',
        texto_limpio,
        flags=re.IGNORECASE
    )

    # 4. Eliminar números de página aislados
    texto_limpio = re.sub(
        r'\n\s*\d+\s*\n',
        '\n',
        texto_limpio)

    # Normalizar artículos (quitar numeración duplicada tipo "(275)")
    texto_limpio = re.sub(
        r'\(\d+\)\s*Artículo',
        'Artículo',
        texto_limpio)

    # Eliminar líneas completamente en blanco múltiples
    texto_limpio = re.sub(
        r'\n{3,}',
        '\n\n',
        texto_limpio)

    # Normalizar espacios
    texto_limpio = re.sub(
        r'[ \t]+',
        ' ',
        texto_limpio)

    return texto_limpio.strip()

def limpiar_ruido_dof(text: str) -> str:
    text = text.replace('\n', ' ')

    # Quitar encabezados institucionales repetitivos de CNBV / SHCP
    patterns_headers = [
        r'CÁMARA DE DIPUTADOS.*?SERVICIOS PARLAMENTARIOS',
        r'LEY DE INSTITUCIONES DE CRÉDITO',
        r'ESTADOS UNIDOS MEXICANOS',
        r'Última Reforma DOF.*?\d{4}'
    ]

    for p in patterns_headers:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    # Remover anotaciones legales
    patterns_dof = [
        r'Párrafo reformado DOF.*?\d{4}',
        r'Artículo reformado DOF.*?\d{4}',
        r'Artículo adicionado DOF.*?\d{4}',
        r'Fracción reformada DOF.*?\d{4}',
        r'Fracción adicionada DOF.*?\d{4}',
        r'Reforma DOF.*?\d{4}',
        r'Derogado DOF.*?\d{4}'
    ]

    for p in patterns_dof:
        text = re.sub(p, '', text)

    # Remover número de páginas
    text = re.sub(r'\d+\s+de\s+\d+', '', text)

    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)

    # Preservar estructura: artículos
    text = re.sub(r'(Artículo\s+\d+\s*(Bis)?\.?)', r'\n\1', text)

    # Preservar números romanos (fracciones)
    text = re.sub(r'\s([IVX]+\.)', r'\n\1', text)

    return text.strip()


def limpiar_ruido_basel(texto: str) -> str:
    """
    Limpia ruido documental típico de documentos del Comité de Basilea/BIS traducidos al español.
    Conserva el contenido normativo principal y elimina portada, avisos, pies de página,
    numeración, ISBN e información editorial.
    """

    texto_limpio = texto

    # Nota de traducción en portada
    texto_limpio = re.sub(
        r"Traducción al español generada.*?Marco de Basilea consolidado\.",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.DOTALL)

    # Encabezado/pie repetido con título del documento
    texto_limpio = re.sub(
        r"Herramientas de monitoreo para la gestión de la liquidez intradía\s*-\s*traducción al español",
        "",
        texto_limpio,
        flags=re.IGNORECASE)

    # Números de página tipo: Página 1 de 16
    texto_limpio = re.sub(
        r"Página\s+\d+\s+de\s+\d+",
        "",
        texto_limpio,
        flags=re.IGNORECASE)

    # Texto editorial de disponibilidad
    texto_limpio = re.sub(
        r"Esta publicación está disponible en el sitio web del BIS.*?\.",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.DOTALL)

    # Copyright / derechos reservados
    texto_limpio = re.sub(
        r"©\s*Banco de Pagos Internacionales\s+2013\..*?cite la fuente\.",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.DOTALL)

    # ISBN
    texto_limpio = re.sub(
        r"ISBN\s+[\d\-Xx]+\s*\(?(impreso|en línea|online|print)?\)?",
        "",
        texto_limpio,
        flags=re.IGNORECASE)

    # Eliminar línea de institución si aparece aislada
    texto_limpio = re.sub(
        r"^\s*Comité de Basilea\s+de Supervisión Bancaria\s*$",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.MULTILINE)

    # Eliminar tabla de contenido / índice con puntos y número de página
    # Ejemplo: "I. Introducción ........................................ 1"
    texto_limpio = re.sub(
        r"^\s*(Contenido|Índice)\s*$",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.MULTILINE)

    texto_limpio = re.sub(
        r"^\s*[IVXLCDM]+\.\s+.*?\.{3,}\s*\d+\s*$",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.MULTILINE)

    texto_limpio = re.sub(
        r"^\s*[A-Z]\.\s+.*?\.{3,}\s*\d+\s*$",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.MULTILINE )

    texto_limpio = re.sub(
        r"^\s*Anexo\s+\d+\..*?\.{3,}\s*\d+\s*$",
        "",
        texto_limpio,
        flags=re.IGNORECASE | re.MULTILINE )

    # Quitar números de página aislados
    texto_limpio = re.sub(
        r"\n\s*\d+\s*\n",
        "\n",
        texto_limpio)

    # Normalizar viñetas: mantenerlas, pero limpiar espacios
    texto_limpio = re.sub(
        r"\n\s*[•●]\s*",
        "\n- ",
        texto_limpio)

    # Normalizar espacios y saltos
    texto_limpio = re.sub(r"[ \t]+", " ", texto_limpio)
    texto_limpio = re.sub(r"\n{3,}", "\n\n", texto_limpio)

    return texto_limpio.strip()




def procesar_documento(texto: str, origen: str = "CNBV") -> str:
    if origen.upper() == "CNBV":
        texto = limpieza_universal(limpiar_ruido_cnbv(texto))
    if origen.upper() == "BANXICO":
        texto = limpieza_universal(limpiar_ruido_banxico(texto))
    if origen.upper() == "DOF":
        texto = limpieza_universal(limpiar_ruido_dof(texto))
    if origen.upper() == "BASEL":
        texto = limpieza_universal(limpiar_ruido_basel(texto))
    return limpieza_universal(texto)
