import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "huggingface_cache")
os.environ["HF_TOKEN"] = "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"
os.environ["LANCEDB_CONFIG_DIR"] = os.path.join(os.getcwd(), "lancedb_config")
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = os.path.join(os.getcwd(), "pytorch_kernel_cache")

if __name__ == "__main__":
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.lancedb import LanceDBVectorStore
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import (
        messages_to_prompt,
        completion_to_prompt,
    )

    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf"

    llm = LlamaCPP(
        model_url=model_url,
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 3},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
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
    response = query_engine.query("What is a CXL type 3 device ? How is it different from a Type 1 device ?")
    print(response)
