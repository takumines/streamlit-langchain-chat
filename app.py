import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

def create_agent_chain():
    chat = ChatOpenAI(
        model_name=os.environ.get("OPENAI_API_MODEL"),
        temperature=os.environ.get("OPENAI_API_TEMPERATURE"),
        streaming=True,
    )
    # OpenAI FUnctions AgentのプロンプトにMemoryの会話履歴を追加する設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]
    }

    # OpenAI Functions Agentが使える設定でMemoryを初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory)

st.title("デモチャット")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
