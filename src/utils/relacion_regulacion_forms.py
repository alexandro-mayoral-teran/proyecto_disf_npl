import Pandas as pd
import re

def contar_presencia_terminos(terminos, texto):
    '''
    Función para contar el número de términos contenidos en un texto
    '''
    resultados = []

    for termino in terminos:
        if pd.isna(termino):
            continue

        termino = str(termino).strip().lower()
        if len(termino) < 3:
            continue

        ocurrencias = len(re.findall(rf"(?<!\w){re.escape(termino)}(?!\w)", texto))

        if ocurrencias > 0:
            resultados.append({
                "termino": termino,
                "ocurrencias": ocurrencias
            })

    return pd.DataFrame(resultados).sort_values("ocurrencias", ascending=False).drop_duplicates().reset_index(drop=True)
