from llama_index.core import load_index_from_storage, StorageContext, SummaryIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.readers.mongodb import SimpleMongoReader
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
import pymongo
from tqdm import tqdm


host = "localhost"
port = 27017
db_name = "default_db"
collection_name = "default_collection"
field_names = ["text"]
query_dict = {}


llm = Ollama(model = "llama3", request_timeout=180)

mongo_uri = "mongodb://localhost:27017"

mongodb_client = pymongo.MongoClient(mongo_uri)

reader = SimpleMongoReader(host, port)

nodes = reader.load_data(
    db_name, collection_name, field_names, query_dict=query_dict
)

retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
nodes = retriever.retrieve("Что такое uml?")
for node in tqdm(nodes):
    print(node)