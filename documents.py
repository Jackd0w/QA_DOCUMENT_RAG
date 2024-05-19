from llama_index.core import VectorStoreIndex, Document, Settings, SimpleDirectoryReader 
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index import SemanticSplitterNodeParser
from llama_index.core.node_parser import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
import numpy as np
import httpx
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
semantic_parser = SemanticSplitterNodeParser(buffer_size=5, embed_model='paraphrase-MiniLM-L6-v2')




documents = SimpleDirectoryReader("Simple_QA_Rag/data").load_data()
print(f'Loaded {len(documents)} docs')
nodes = semantic_parser.from_documents(documents)

embeddings = model.encode(nodes)
d = embeddings.shape[1]  # Размерность эмбеддингов
index = faiss.IndexFlatL2(d)
print("Database created.")
index.add(embeddings)

index.storage_context.persist(persist_dir="Simple_QA_Rag/vec_data")


"""
vector_store = FaissVectorStore(len(nodes))

for node in nodes:
    vector_store.add_node(node)
    
print("Данные успешно загружены и индексированы в Vespa.")
"""