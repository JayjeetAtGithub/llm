import os
import torch

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "huggingface_cache")
os.environ["HF_TOKEN"] = "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"
os.environ["LANCEDB_CONFIG_DIR"] = os.path.join(os.getcwd(), "lancedb_config")
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = os.path.join(os.getcwd(), "pytorch_kernel_cache")

if __name__ == "__main__":
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, SimpleInputPrompt
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.lancedb import LanceDBVectorStore

    system_prompt = """
        You are a Q/A assistant for a research paper library. Your goal is to answer questions 
        as accurately as possible based on the instructions and context provided.  
    """
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    model = "meta-llama/Llama-2-7b-hf"
    
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
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

    query = str(input("Enter query: "))

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
