import Langchain_codes as lc
import streamlit as st
import Phidata_codes as pc
from langchain_core.messages import HumanMessage,AIMessage


st.title("Personal Finance Chat Bot 💻💼")

with st.sidebar:
    st.subheader("Select Framework")
    if "agent" not in st.session_state:
        st.session_state.agent = None  # Default is None
    col1, col2 = st.columns(2)  # Two buttons side by side
    with col1:
        if st.button("🔹 Langchain"):
            if(st.session_state.agent!='Langchain'):
                st.session_state.messages=[]
            st.session_state.agent = "Langchain"

    with col2:
        if st.button("🔹 PhiData"):
            if (st.session_state.agent != 'PhiData'):
                st.session_state.messages = []
                # pc.memory.clear()
            st.session_state.agent = "PhiData"

if "messages" not in st.session_state:
    st.session_state.messages=[]


with st.sidebar:
    if(st.session_state.agent is None):
        st.subheader("Select a framework")

    elif(st.session_state.agent=='Langchain'):
        st.subheader("Using Langchain Framework For Answering Questions")

    elif(st.session_state.agent=='PhiData'):
        st.subheader("Using Phidata Framework For Answering Questions")

for m in st.session_state.messages:
    if(isinstance(m,HumanMessage)):
        with st.chat_message('user'):
            st.markdown(m.content)

    elif(isinstance(m,AIMessage)):
        with st.chat_message('assistant'):
            st.markdown(m.content)


user_message=st.chat_input("Hello, How are you?")

if(user_message):
    if(st.session_state.agent is None):
        with st.chat_message('assistant'):
            st.markdown('Please Select A Framework')

    if(st.session_state.agent is not None):
        with st.chat_message('user'):
            st.markdown(user_message)
            st.session_state.messages.append(HumanMessage(user_message))

        with st.spinner('Generating Response'):
            if(st.session_state.agent=='Langchain'):
                response=lc.get_response(user_message,st.session_state.messages)
                if(response):
                    response=response['output']
            else:
                response=pc.get_response(user_message).content
            with st.chat_message('assistant'):
                st.write(response)
                st.session_state.messages.append(AIMessage(response))
