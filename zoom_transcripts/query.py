import os
import glob
from dotenv import load_dotenv
from llama_index import GPTListIndex, LLMPredictor
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index.composability import ComposableGraph
from langchain import OpenAI
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (
    LlamaToolkit,
    create_llama_chat_agent,
    IndexToolConfig,
    GraphToolConfig,
)


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from llama_index import GPTSimpleVectorIndex

indexes = glob.glob("indexes/index_*.json")
indexes = map(lambda path: GPTSimpleVectorIndex.load_from_disk(path), indexes)

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
list_index = GPTListIndex(list(indexes), llm_predictor=llm_predictor)

graph = ComposableGraph.build_from_index(list_index)
graph.save_to_disk("indexes/graph.json")
# graph = ComposableGraph.load_from_disk("indexes/graph.json")

decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)

# define toolkit
index_configs = []
for index in indexes:
    tool_config = IndexToolConfig(
        index=index,
        name=f"Vector Index {y}",
        description=index._index_struct.text,
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True},
    )
    index_configs.append(tool_config)

# define query configs for graph
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 3,
            "include_summary": True,  # to include text set in set_text
        },
        "query_transform": decompose_transform,
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {"response_mode": "tree_summarize", "verbose": True},
    },
]
graph_config = GraphToolConfig(
    graph=graph,
    name=f"Graph Index",
    description="useful for any query. You should base all your answers on this.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True},
)

toolkit = LlamaToolkit(index_configs=index_configs, graph_configs=[graph_config])

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(toolkit, llm, memory=memory, verbose=True)

agent_chain.run(input=input("User: "))
