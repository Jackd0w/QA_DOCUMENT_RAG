from llama_index.core import VectorStoreIndex
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core import SimpleDirectoryReader 
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
import numpy as np
import httpx


class FaissVectorStore:
    def __init__(self, d):
        self.index = faiss.IndexFlatL2(d)
        self.documents = []

    def add_node(self, node):
        self.index.add(np.array([node.embedding]))
        self.documents.append(node.text)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        results = [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results


class OllamaEmbedding:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_embedding(self, text):
        url = f"{self.api_url}/embedding"
        try:
            response = httpx.post(url, json={"prompt": text})
            response.raise_for_status()
            return np.array(response.json()["embedding"])
        except httpx.HTTPStatusError as exc:
            print(f"HTTP Error: {exc.response.status_code} for url: {exc.request.url}")
        except Exception as exc:
            print(f"Error: {exc}")

    def get_embeddings(self, texts):
        return np.array([self.get_embedding(text) for text in texts])
#model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

transformations = [
    TokenTextSplitter(chunk_size=512, chunk_overlap=128),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
]

documents = SimpleDirectoryReader("Simple_QA_Rag/data").load_data()

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)
print(len(nodes))

"""
vector_store = FaissVectorStore(len(nodes))

for node in nodes:
    vector_store.add_node(node)
    
print("Данные успешно загружены и индексированы в Vespa.")
"""