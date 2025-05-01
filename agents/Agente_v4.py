# ------------------------------
# 0. Imports y Setup
# ------------------------------
from dotenv import load_dotenv
load_dotenv()

import subprocess
from typing import TypedDict, Annotated, Union

from pydantic import BaseModel
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.agents import AgentAction, AgentFinish
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langgraph.graph import StateGraph, END, START
#from langchain.agents.format_scratchpad.react import format_to_messages as format_react_scratchpad
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.utils.function_calling import convert_to_openai_function
from tools.Tools import tools
from memory.memory_setup import retrieve_relevant_memories , save_long_term_memory

from typing import TypedDict, Annotated, Sequence, List
import operator
from langchain_core.messages import BaseMessage


#Creamos las funciones que serán los nodos del grafo
from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage

# # ------------------------------
# 1. Tools
# # ------------------------------

tool_executor = ToolExecutor(tools)

functions = [convert_to_openai_function(t) for t in tools] #Convertimos las herramientas en un formato compatible con las funciones de OpenAI

# ------------------------------
# 2. Memoria, LLM y Prompt
# ------------------------------

from langchain.memory import ConversationBufferMemory

#Funcion para leer .txt
def load_prompt(filepath: str) -> str:
    """Carga el system message desde un archivo de texto y asegura que sea str."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    if not isinstance(content, str):
        raise ValueError(f"Contenido del archivo {filepath} no es un string válido.")
    return content



memory = ConversationBufferMemory(return_messages=True)
#memory = ChatMessageHistory()

llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = llm.bind_functions(functions)

# Cargar el system message desde archivo
system_message_content = load_prompt("./system_msg/system_msg_v0.txt")

print(f"Contenido cargado: {system_message_content}")
print(f"Tipo de dato cargado: {type(system_message_content)}")


# Crear el system message
system_message = SystemMessage(content=system_message_content)


prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("system", "Herramientas disponibles:\n{tools}\nPuedes usar: {tool_names}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# ------------------------------
# 3. Crear agente
# ------------------------------



# ------------------------------
# 4. Estado del LangGraph
# ------------------------------

class AgentState(TypedDict):
    #No necesitamos etapas intermedias aqui
    messages: Annotated[Sequence[BaseMessage], operator.add] #La clave "messages", debe ser una Secuencia de Mensajes
    recall_memories: Annotated[List[str], operator.add] #Parte del estado que transporta información de memoria a largo plazo
    user_name: str

# ------------------------------
# 5. Nodos del grafo
# ------------------------------

# Nodo Agente Principal - Considera system message
def run_agent(state):
    # Extraemos todos los mensajes hasta ahora
    messages = state["messages"]
    
    #Nos traemos los recuerdos recuperados si es que los hay
    recall_memories = state.get("recall_memories", [])
    # Construimos contexto de recuerdos (si hay)
    memories_context = "\n".join(recall_memories) if recall_memories else ""
    print("Las Memorias del usuario son: " + memories_context)

    #Usuario
    user_name = state["user_name"]

    # Separamos los mensajes en historial y última entrada
    chat_history = memory.load_memory_variables({})["history"]
    last_human = messages[-1]  # debe ser HumanMessage

    # Renderizamos el prompt dinámico
    prompt_input = prompt.invoke({
        "chat_history": chat_history,
        "input": f"""Información del usuario:
                     - Nombre de usuario: {user_name}
                     - Memorias relevantes:
                    {memories_context}

                     ---
                     Nueva instrucción del usuario:
                     {last_human.content}
                     """,
        "tools": "\n".join([t.description for t in tools]),
        "tool_names": ", ".join([t.name for t in tools]),
        "agent_scratchpad": [],  # puede mejorarse con memoria interna si quieres
    })


    #print("El prompt de input es:")
    #print(prompt_input.pretty_print())

    response = llm.invoke(prompt_input)
    return {"messages": [response]}


# 5.2 Nodo que ejecuta herramienta si corresponde
#Herramienta
def execute_tool(state):
    messages = state['messages']
    last_message = messages[-1] #Extraemos el último mensaje del diccionario

    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"], 
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    print(f"The agent action is {action}")

    response = tool_executor.invoke(action)
    print(f"The tool result is: {response}")

    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}

# 5.3 Nodo Condicional , decide entre utilizar la tool o finalizar y guardar en memoria ram
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1] #Extraemos el último mensaje del diccionario
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"


# Nodo de actualización de memoria
def update_memory(state):
    messages = state["messages"]
    for msg in messages:
        memory.chat_memory.add_message(msg)  # Usa el método adecuado de ConversationBufferMemory
    return {"messages": messages}


# Nodo para recuperar memorias relevantes
def retrieve_memory(state: AgentState) -> AgentState:
    messages = state['messages']
    last_message = messages[-1]

    # Aquí debes decidir cómo obtener el user_name
    user_name = "Luis"  # ← DE MOMENTO puedes dejarlo fijo para testear

    # Recuperamos memorias relevantes filtradas por usuario
    retrieved_memories = retrieve_relevant_memories(
        query=last_message.content,
        user_name=user_name,
        top_k=3  # Puedes parametrizar si quieres
    )

    if retrieved_memories:
        return {
            "messages": [],
            "recall_memories": retrieved_memories
        }
    else:
        return {
            "messages": [],
            "recall_memories": []
        }



# ------------------------------
# 6. Definición del grafo
# ------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("tool", execute_tool)
workflow.add_node("retrieve_memory", retrieve_memory)
workflow.add_node("memory", update_memory)  

workflow.add_edge(START, "retrieve_memory")
workflow.add_edge("retrieve_memory", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool",
        "end": "memory"  # <--- antes de terminar, pasamos por la memoria
    }
)

workflow.add_edge("tool", "agent")

workflow.set_finish_point("memory")
workflow.add_edge("memory", END)


graph_executor = workflow.compile()

