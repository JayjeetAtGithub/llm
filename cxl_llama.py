import os
import logging
import sys
import torch
import textwrap

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "huggingface_cache")
    os.environ["HF_TOKEN"] = "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"
    os.environ["LANCEDB_CONFIG_DIR"] = os.path.join(os.getcwd(), "lancedb_config")

    model = "meta-llama/Llama-2-7b-hf"

    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    """

    prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )
    print(prompt)

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        query_wrapper_prompt=prompt,
        tokenizer_name=model,
        model_name=model,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": False},
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader("papers_data").load_data()

    vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(textwrap.fill(str(response), 100))
