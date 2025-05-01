import pygsheets #pip install pygsheets
import pandas as pd #pip instal pandas

from pydantic import BaseModel

from langchain.tools import Tool, StructuredTool

# Modelo con campos esperados
class CamposBase(BaseModel):
    Fecha: str
    Zona: str
    Actividad: str
    Planificado: float
    Ejecutado: float
    Estado: str
    Observaciones: str


# Función que recibe el modelo
def registrar_google_sheets(data: CamposBase) -> bool:
    try:
        # Leer datos existentes desde Google Sheets
        sheet_id = "1Iw15CIQ6bzmS6ozM3JdEAt6toEUhMaZ46eu7vXnvvdo"
        sheet_name = "leads"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(csv_url)

        # Agregar la nueva fila desde el modelo
        nueva_fila = {
            "Fecha": data.Fecha,
            "Zona": data.Zona,
            "Actividad": data.Actividad,
            "Planificado": data.Planificado,
            "Ejecutado": data.Ejecutado,
            "Estado": data.Estado,
            "Observaciones": data.Observaciones
        }

        df.loc[len(df.index)] = nueva_fila

        # Autorizar con pygsheets y subir la nueva versión del DataFrame
        service_account_path = 'utils/chat-streamlit-deploy-luis-e3b98b293457.json'
        gc = pygsheets.authorize(service_file=service_account_path)

        editable_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid=0"
        sh = gc.open_by_url(editable_url)
        wks = sh[0]
        wks.set_dataframe(df, (1, 1))

        return True

    except Exception as e:
        print(f"Error al registrar en Google Sheets: {e}")
        return False

# StructuredTool para LangChain
registrar_google_sheet = StructuredTool.from_function(
    func=registrar_google_sheets,
    name="registrar_google_sheet",
    description="Registra un avance de obra con campos como Fecha, Zona, Actividad, Planificado, Ejecutado, Estado y Observaciones en Google Sheets.",
    args_schema=CamposBase
)

