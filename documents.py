import pymongo
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

mongo_uri = "mongodb://localhost:27017"


mongodb_client = pymongo.MongoClient(mongo_uri)
store = MongoDBAtlasVectorSearch(mongodb_client)
storage_context = StorageContext.from_defaults(vector_store=store)


embed_model = HuggingFaceEmbedding(
    model_name="cointegrated/rubert-tiny"
)


documents = SimpleDirectoryReader("Simple_QA_Rag/data").load_data()
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, embed_model = embed_model, storage_context=storage_context)

retriever = BM25Retriever.from_defaults(nodes=documents, similarity_top_k=2)

#vector_index.as_query_engine(embed_model)
nodes = retriever.retrieve("Какие предметы упоминаются в документе?")
for node in nodes:
    print(node)