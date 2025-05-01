# ------------------------------
# 0. Imports y Setup
# ------------------------------
from dotenv import load_dotenv
load_dotenv()

import subprocess
from typing import TypedDict, Annotated, Union

from pydantic import BaseModel
from langchain.tools import Tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.agents import AgentAction, AgentFinish
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langgraph.graph import StateGraph, END
#from langchain.agents.format_scratchpad.react import format_to_messages as format_react_scratchpad
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.utils.function_calling import convert_to_openai_function



# ------------------------------
# 1. Tool: Execute CMD
# ------------------------------

class ExecuteCmdInput(BaseModel):
    command: str

def execute_cmd_tool(input: Union[str, dict, ExecuteCmdInput]) -> str:
    print(f"🔍 Tipo de input recibido: {type(input)}")
    print(f"📦 Input crudo: {input}")
    try:
        if isinstance(input, str):
            command = input
        elif isinstance(input, dict):
            command = input.get("command")
        elif isinstance(input, ExecuteCmdInput):
            command = input.command
        else:
            return "❌ Tipo de input no reconocido"

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"❌ Error al ejecutar el comando: {str(e)}"

execute_cmd_tool_instance = Tool(
    name="execute_cmd",
    description="Ejecuta un comando del sistema operativo en la terminal CMD y devuelve la salida.",
    func=execute_cmd_tool,
    args_schema=ExecuteCmdInput
)


# # --------------------------------
# Tool 2 : create_file
# # --------------------------------

class CreateFileInput(BaseModel):
    filename: str
    content: str

def create_file_tool(filename: str, content: str) -> str:
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"✅ Archivo '{filename}' creado correctamente."
    except Exception as e:
        return f"❌ Error al crear el archivo: {str(e)}"

create_file_tool_instance = StructuredTool(
    name="create_file",
    description="Crea un archivo con nombre y contenido personalizado. Útil para generar scripts, YAMLs o configuraciones.",
    func=create_file_tool,
    args_schema=CreateFileInput
)



# # --------------------------------
# Tool 3 : read_file
# # --------------------------------

class ReadFileInput(BaseModel):
    filename: str

def read_file_tool(filename: str) -> str:
    try:
        with open(filename, "r") as f:
            content = f.read()
        return content if content else "⚠️ El archivo está vacío."
    except FileNotFoundError:
        return f"❌ El archivo '{filename}' no existe."
    except Exception as e:
        return f"❌ Error al leer el archivo: {str(e)}"

read_file_tool_instance = Tool(
    name="read_file",
    description="Lee y devuelve el contenido completo de un archivo. Úsala para revisar scripts, .env, YAMLs o configuraciones antes de modificarlas.",
    func=read_file_tool,
    args_schema=ReadFileInput
)


from memory.memory_setup import save_long_term_memory_tool

tools = [execute_cmd_tool_instance, create_file_tool_instance, read_file_tool_instance, save_long_term_memory_tool]
