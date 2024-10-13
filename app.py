#!pip install langchain langchain-community langchain-openai openai tiktoken chromadb langchain_chroma streamlit

import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
def RAG(input_question):
    llm_model = ChatOpenAI(
        openai_api_key = openai_api_key,
        model="gpt-4o-mini",
    )
    persist_directory = "db"

    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding
    )
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1}
    )  # search_kwargs={"k": 3,"score_threshold": 0.6}

    #################################
    prompt_template = """You are an professional encouraging doctor who helps patients.
    Answer the questions using the facts provided. You can ask further question for more information.
    List your answer with each element adding new lines between the list.
    If you don't know the answer, don't try to make up answers. Ask questions to get more information, or ask the patient to ask for professional advice from doctor.
    You will answer in mandarin only.
    {summaries} 
    """
    
    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain.invoke(input_question)
    

    return result["answer"]


assistant_logo = 'https://cdn.pixabay.com/photo/2021/11/09/05/44/doctor-6780685_1280.png'
st.set_page_config(
    page_title="Colonoscopy Assistant",
    page_icon=assistant_logo
)
st.markdown("## 台北市立聯合醫院仁愛院區 健康管理中心")
st.markdown("## 大腸鏡衛教機器人")
st.markdown("歡迎來到 台北市立聯合醫院仁愛院區 健康管理中心 進行:red[**大腸鏡檢查**]。")
st.markdown(":white_circle: 您可以先參考注意事項影片 :arrow_forward: https://www.youtube.com/watch?v=z1YI8oimJto 。")
st.markdown(":white_circle: 也可以直接詢問我關於檢查的注意事項。")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": """您好，歡迎來到台北市立聯合醫院仁愛院區進行大腸鏡檢查。請問我可以提供什麼協助嗎？"""}]

for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=assistant_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if query := st.chat_input("輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in RAG(query):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
