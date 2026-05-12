import Pandas as pd

def resumen_formulario(formularios: dict) -> pd.DataFrame:
    '''
    Función que resume características asociadas al formulario.
    formularios: Diccionario de tablas (secciones) que conforman el formulario.

    Devuelve:
    Dataframe con la información resumida.
    '''
    filas = []

    for seccion, df in formularios.items():
        columnas_catalogo = [c for c in df.columns if "catalogo" in c]

        filas.append({
            "seccion": seccion,
            "n_campos": len(df),
            "n_columnas_definicion": df.shape[1],
            "campos_con_descripcion": df["descripcion"].notna().sum() if "descripcion" in df.columns else np.nan,
            "campos_con_catalogo": df[columnas_catalogo].notna().any(axis=1).sum() if columnas_catalogo else 0,
            "campos_llave": df["llave"].notna().sum() if "llave" in df.columns else 0
        })

    return pd.DataFrame(filas)

def resumen_catalogos(catalogos: dict) -> pd.DataFrame:
    '''
    Función que resume características asociadas a los catálogos de un formulario.
    catalogos: Diccionario de tablas (catálogos) asociados a un formulario.

    Devuelve:
    Dataframe con la información resumida.
    '''
    filas = []

    for nombre, df in catalogos.items():
        filas.append({
            "catalogo": nombre,
            "n_registros": len(df),
            "n_columnas": df.shape[1],
            "columnas": ", ".join(df.columns.astype(str))
        })

    return pd.DataFrame(filas)

def missing_por_seccion(formularios: dict) -> pd.DataFrame:
    '''
    Función que muestra el número de valores faltantes por campo, según la sección del formulario.
    formularios: Diccionario de tablas (secciones) que conforman el formulario.

    Devuelve:
    Dataframe con la información resumida.
    '''
    filas = []

    for seccion, df in formularios.items():
        for col in df.columns:
            filas.append({
                "seccion": seccion,
                "columna": col,
                "n_faltantes": df[col].isna().sum(),
                "porcentaje_faltantes": df[col].isna().mean()
            })

    return pd.DataFrame(filas)

def extraer_relacion_seccion_catalogo(formularios: dict) -> pd.DataFrame:
    '''
    Función que verifica los catálogos asociados a cada campo y sección de los formularios.
    '''
    relaciones = []

    for seccion, df in formularios.items():
        columnas_catalogo = [c for c in df.columns if "catalogo" in c]

        for _, row in df.iterrows():
            campo = row.get("etiqueta", None)

            for col_cat in columnas_catalogo:
                catalogo = row.get(col_cat, None)

                if pd.notna(catalogo):
                    relaciones.append({
                        "seccion": seccion,
                        "campo": campo,
                        "columna_catalogo": col_cat,
                        "catalogo": str(catalogo).strip()
                    })

    return pd.DataFrame(relaciones)

def construir_dataset_campos(formularios: dict) -> pd.DataFrame:
    registros = []

    for seccion, df in formularios.items():
        columnas_catalogo = [c for c in df.columns if "catalogo" in c]

        for idx, row in df.iterrows():
            catalogos_asociados = [
                str(row[c]).strip()
                for c in columnas_catalogo
                if c in row and pd.notna(row[c])
            ]

            texto_campo = " | ".join([
                str(row.get("etiqueta", "")),
                str(row.get("descripcion", "")),
                str(row.get("tipo_dato", "")),
                " ".join(catalogos_asociados)
            ])

            registros.append({
                "tipo_documento": "formulario",
                "seccion": seccion,
                "campo": row.get("etiqueta", None),
                "descripcion": row.get("descripcion", None),
                "tipo_dato": row.get("tipo_dato", None),
                "catalogos_asociados": catalogos_asociados,
                "n_catalogos_asociados": len(catalogos_asociados),
                "texto_representacion": texto_campo,
                "n_caracteres": len(texto_campo),
                "n_palabras": len(re.findall(r"\b\w+\b", texto_campo))
            })

    return pd.DataFrame(registros)
