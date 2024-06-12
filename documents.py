import pymongo
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core.schema import MetadataMode
from tqdm import tqdm

mongo_uri = "mongodb://localhost:27017"


mongodb_client = pymongo.MongoClient(mongo_uri)
store = MongoDBAtlasVectorSearch(mongodb_client)
storage_context = StorageContext.from_defaults(vector_store=store)


Settings.llm = Ollama(model="llama2", request_timeout=60)
Settings.embed_model = HuggingFaceEmbedding(
        model_name="cointegrated/rubert-tiny")


documents = SimpleDirectoryReader("Simple_QA_Rag/data").load_data()

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, transformations = [
        TokenTextSplitter(chunk_size=128, chunk_overlap=32),
        HuggingFaceEmbedding(
        model_name="cointegrated/rubert-tiny")
    ], show_progress=True)
