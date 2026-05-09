import re

def segmentar_texto(texto: str) -> list:
    '''
    Función para dividir el texto en segmentos a partir de líneas en blanco, es decir, cada segmento sería un párrafo.
    '''
    segmentos = [bloque.strip() for bloque in re.split(r"\n\s*\n", texto) if bloque.strip()]
    return segmentos

def estadisticas_segmento(texto: str) -> dict:
    '''
    Función para mostrar estadísticas encontradas en cada segmento.
    '''
    palabras = re.findall(r"\b\w+\b", texto, flags=re.UNICODE)
    return {
        "segmento": texto,
        "n_caracteres": len(texto),
        "n_palabras": len(palabras),
        "n_lineas": len(texto.splitlines()),
        "longitud_promedio_palabra": round(
            sum(len(p) for p in palabras) / len(palabras), 2
        ) if palabras else 0
    }
