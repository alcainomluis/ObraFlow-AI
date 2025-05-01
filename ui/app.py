import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from agents.Agente_v4 import graph_executor  # Asegúrate de importar bien tu agente compilado

# Configuración básica
st.set_page_config(page_title="Agente MLOps", page_icon="🤖")

st.title("🧠 Agente MLOps Terminal")

# Chat histórico básico
if "history" not in st.session_state:
    st.session_state.history = []

# Input del usuario
user_input = st.chat_input("Escribe una instrucción para el agente...")

# Mostrar historial
for i, (role, message) in enumerate(st.session_state.history):
    with st.chat_message(role):
        st.markdown(message)

# Procesar entrada
if user_input:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append(("user", user_input))

    # Preparamos el input del usuario (AGENT STATE CORRECTO)
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "recall_memories": []
    }

    # Invocar agente
    with st.spinner("Pensando..."):
        result = graph_executor.invoke(inputs)

    # Extraer respuesta final (último AIMessage)
    final_response = None
    for msg in reversed(result["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # Mostrar respuesta del agente
    with st.chat_message("assistant"):
        st.markdown(final_response)
    st.session_state.history.append(("assistant", final_response))
