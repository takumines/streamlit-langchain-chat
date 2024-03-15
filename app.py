import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

st.title("デモチャット")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat = ChatOpenAI(
            model_name=os.environ.get("OPENAI_API_MODEL"),
            temperature=os.environ.get("OPENAI_API_TEMPERATURE"),
        )

        messages = [HumanMessage(content=prompt)]
        response = chat.invoke(messages)
        content = response.content
        st.markdown(content)
        st.session_state.messages.append({"role": "assistant", "content": content})