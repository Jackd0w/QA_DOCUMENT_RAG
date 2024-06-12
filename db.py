from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent


def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = Ollama(model="Llama3")

agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)