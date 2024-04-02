from llama_index.core import PromptTemplate
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings, VectorStoreIndex

selected_model = r"D:\Qwen1.5-4B-Chat"

SYSTEM_PROMPT = """You are a helpful AI assistant.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    # query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16},
)
Settings.llm = llm

from llama_index.core.readers import download_loader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:/BAAI/bge-base-zh-v1.5"
)
# Settings.embed_model = r"local:D:\BAAI\bge-base-zh-v1.5"
# Setting.chunk_size = 512

# 读文档
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=f'D:\Milvus_test\chatgpt_rag_test\华为OceanProtect备份一体机产品技术白皮书.pdf')
print(len(documents))
print(documents[0])

# 文档拆分为块/节点
# create nodes parser
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
# split into nodes
nodes = node_parser.get_nodes_from_documents(documents)
print(len(nodes))
print(nodes[0])

# vector_index = VectorStoreIndex(nodes)
# #持久化向量存储
# vector_index.storage_context.persist(f'D:\Milvus_test\chatgpt_rag_testindex3')

from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="../chatgpt_rag_testindex3")
vector_index = load_index_from_storage(storage_context=storage_context)

# 创建检索器、并把检索器插入查询引擎
query_engine = vector_index.as_query_engine()
print(1)
# 测试结果
response_vector = query_engine.query("OceanProtect是什么?")
print(response_vector)
response_vector = query_engine.query("OceanProtect X3000最大节点数量?")
print(response_vector.response)
