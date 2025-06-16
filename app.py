import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

# Load GROQ and OPEANAI API keys
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("RAG Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate responsebased on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """
)


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama2")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  ##Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  ## Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        )  ## splitting and chunk creation
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )  ## creating vectors


prompt1 = st.text_input("Enter your question from documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector store DB is Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relavant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------")
