from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.schema import MetadataMode
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import os
import PyPDF2
import string

CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/Document_RAG"
COLLECTION_NAME = "state"
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding=HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny")

def create_retriever(db_path, embedding):
        loader=PyPDFLoader("data/bellstoll.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
        vectorstore.save_local(db_path)
    
def vectorise(file_name, db):
        loader=PyPDFLoader("data/bellstoll.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
    
def similarity_search(user_query, db):
        return db.similarity_search(user_query)
    
    
    
if __name__ == "__main__":
        #create_retriever(DB_FAISS_PATH, embedding)
        db = FAISS.load_local("vectorstore/db_faiss/", embedding, allow_dangerous_deserialization=True)

        print(similarity_search("Who was the leader of Spanish nationalits", db)[0])