from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# ESQUEMAS PARA CATÁLOGOS
# ==========================================
class ValorCatalogo(BaseModel):
    clave: str = Field(..., description="Identificador o clave del valor (ej. '01', 'MXN', 'ACT').")
    descripcion: str = Field(..., description="Descripción del valor (ej. 'Pesos Mexicanos', 'Activo').")

class CatalogoAsociado(BaseModel):
    nombre_catalogo: str = Field(..., description="Nombre descriptivo del catálogo (ej. 'Catálogo de Monedas').")
    valores: List[ValorCatalogo] = Field(..., description="Lista de valores cerrados permitidos para este catálogo.")

# ==========================================
# ESQUEMAS PARA EL FORMULARIO
# ==========================================
class CampoFormulario(BaseModel):
    nombre_campo: str = Field(..., description="El nombre técnico o identificador del campo para la base de datos.")
    tipo_dato: str = Field(..., description="El tipo de dato esperado (ej. Numérico, Texto, Fecha, Booleano).")
    longitud: Optional[str] = Field(None, description="La longitud máxima del campo si la regulación lo especifica (ej. '10', 'Ilimitado').")
    descripcion_funcional: str = Field(..., description="Descripción detallada de lo que representa el campo según la regulación.")
    formula_calculo: Optional[str] = Field(None, description="Si el campo es calculado (ej. PI, Reservas), incluye aquí la fórmula matemática o lógica descrita en la norma.")
    es_catalogo: bool = Field(..., description="Indica si este campo requiere una lista cerrada de valores (Catálogo).")
    nombre_catalogo_vinculado: Optional[str] = Field(None, description="Si es_catalogo es True, indicar el nombre del catálogo correspondiente.")
    validaciones_sugeridas: Optional[List[str]] = Field(None, description="Reglas de negocio o validaciones cruzadas deducidas del texto (ej. 'No puede ser negativo', 'Debe ser menor a la fecha actual').")

# ==========================================
# ESQUEMA PRINCIPAL (CONTRATO DE SALIDA DEL LLM)
# ==========================================
class RequerimientoInformacion(BaseModel):
    """
    Este es el esquema principal que engloba toda la salida.
    Al forzar al LLM a responder con este esquema, garantizamos que siempre
    devuelva la estructura tabular y los catálogos en un JSON predecible.
    """
    nombre_formulario: str = Field(..., description="Nombre propuesto para el requerimiento o formulario basado en el documento.")
    campos_formulario: List[CampoFormulario] = Field(..., description="Lista de todas las columnas/campos que componen el formulario.")
    catalogos_identificados: List[CatalogoAsociado] = Field(default_factory=list, description="Lista de catálogos encontrados con sus respectivos valores.")
    ambiguedades_detectadas: Optional[List[str]] = Field(None, description="Cualquier contradicción o falta de claridad detectada en la normativa original.")
