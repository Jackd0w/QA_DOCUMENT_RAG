from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch



vector_store = FaissVectorStore.from_persist_dir("Simple_QA_Rag/vec_data")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="Simple_QA_Rag/vec_data"
)

index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Who is the main character?")
streaming_response.print_response_stream()