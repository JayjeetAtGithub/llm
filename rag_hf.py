import os
import torch

# Setup the local caches for HuggingFace, LanceDB and PyTorch
if not os.path.exists("huggingface_cache"):
    os.makedirs("huggingface_cache")

if not os.path.exists("lancedb_config"):
    os.makedirs("lancedb_config")

if not os.path.exists("pytorch_kernel_cache"):
    os.makedirs("pytorch_kernel_cache")

# Set the environment variables for the caches
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "huggingface_cache")
os.environ["HF_TOKEN"] = "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"
os.environ["LANCEDB_CONFIG_DIR"] = os.path.join(os.getcwd(), "lancedb_config")
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = os.path.join(os.getcwd(), "pytorch_kernel_cache")

if __name__ == "__main__":
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
    from llama_index.core.prompts.prompts import SimpleInputPrompt
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.lancedb import LanceDBVectorStore

    # Define the system and query wrapper prompts
    system_prompt = """
        You are a Q/A assistant for a research paper library. Your goal is to answer questions 
        as accurately as possible based on the instructions and context provided.  
    """
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    model_to_use = str(input("Enter the Llama model to use (7b/13b): "))
    if model_to_use == "7b":
        model = "meta-llama/Llama-2-7b-chat-hf"
    elif model_to_use == "13b":
        model = "meta-llama/Llama-2-13b-chat-hf"
    else:
        print("Invalid model specified. Exiting...")
        exit(1)
    
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model,
        model_name=model,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": False},
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Set the LLM and Embedding models in the settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    # Load the documents and create the index
    documents = SimpleDirectoryReader("papers_data").load_data()
    vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

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
        exit(0)
