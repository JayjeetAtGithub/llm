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

    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf"

    print(messages_to_prompt)
    print(completion_to_prompt)

    llm = LlamaCPP(
        model_url=model_url,
        model_path=None,
        temperature=0.0,
        max_new_tokens=2048,
        context_window=4096,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 3},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader("dataset_1").load_data()

    vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    query_engine = index.as_query_engine()
    # Query the index
    try:
        while True:
            query = str(input("Enter query: "))
            if len(query) > 0:
                query_engine = index.as_query_engine()
                response = query_engine.query(query)
                print(response)
            else:
                print("No query provided !")
    except KeyboardInterrupt:
        print("Exiting...")
