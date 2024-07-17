import streamlit as st
from langchain.chains import RetrievalQA
from llama_index.llms.ollama import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os
from documents import embedding, create_retriever
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load and prepare documents
def load_documents():
    document_texts = ["Doc 1: This is a sample document.", "Doc 2: Another example document."]
    return document_texts

# Initialize embeddings and FAISS index
def open_faiss():
    db = FAISS.load_local("vectorstore/db_faiss/", embedding, allow_dangerous_deserialization=True)
    return db

# Initialize LangChain RetrievalQA chain
def initialize_qa_chain(db):
    llm = Ollama(model="Llama3")
    combine_documents_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
    qa_chain = RetrievalQA(retriever=db.as_retriever(), combine_documents_chain=combine_documents_chain)
    return qa_chain



model = Ollama(model="llama3")
# Function to generate response from the model
def generate_response(question):
    #inputs = embedding.encode(question, return_tensors='pt')
    outputs = model.generate(question, max_length=512, num_return_sequences=1)
    #response = embedding.decode(outputs[0], skip_special_tokens=True)
    return outputs

# Streamlit interface
st.title("Q&A with LangChain RAG")
st.write("Ask a question and get answers from the document corpus")

if 'new_kb_name' not in st.session_state:
    st.session_state.new_kb_name = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

st.sidebar.title("Knowledge Base Selection")
knowledge_bases = ["Матан", "CV", "Хэмингуей"]
selected_knowledge_base = st.sidebar.selectbox("Select a Knowledge Base", knowledge_bases)


st.sidebar.write("### Create New Knowledge Base")

if st.sidebar.button("Create New Knowledge Base"):
    new_kb_name = st.sidebar.text_input("Enter the name of the new Knowledge Base")
    #create_retriever("vector_store/db_faiss" + new_kb_name, embedding)
    st.sidebar.write("### Upload Files for New Knowledge Base")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True, key="uploaded_files")
    if new_kb_name and uploaded_files:
        st.sidebar.success(f"Knowledge Base '{new_kb_name}' created successfully with {len(uploaded_files)} files.")
        st.session_state.new_kb_name = ""
        st.session_state.uploaded_files = []
        st.session_state.knowledge_bases.append(new_kb_name)
        st.sidebar.write(f"New Knowledge Base Name: {new_kb_name}")
        for uploaded_file in uploaded_files:
            st.sidebar.write(f"Uploaded file: {uploaded_file.name}")


question = st.text_input("Enter your question:")

if question:
    with st.spinner("Generating response..."):
        answer = generate_response(question)
        st.write("**Answer:**")
        st.write(answer)
        st.write("Retrieved documents will be displayed here.")
