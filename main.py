from llama_index.core import TreeIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama



parser = LlamaParse(result_type="markdown")

documents = SimpleDirectoryReader("./data").load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)


llm = Ollama(model="phi", request_timeout=30.0)


query_engine = vector_index.as_query_engine()
response = query_engine.query("Who is the author?")
print(response)