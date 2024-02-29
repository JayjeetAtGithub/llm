import os
from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)


if __name__ == "__main__":
    model_url = "https://huggingface.co/TheBloke/CodeLlama-34B-GGUF/resolve/main/codellama-34b.Q5_K_M.gguf"
    llm = LlamaCPP(
        model_url=model_url,
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Set the LLM and Embedding models in the settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    folder_path = "./storage"
    folder_exists = os.path.exists(folder_path) and os.path.isdir(folder_path)

    if folder_exists:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    else:
        reader = GithubRepositoryReader("JayjeetAtGithub", "llm", ignore_directories=[".github", ".vscode", "benchmarks", "docs", "examples", "experimental", "scripts", "tests"])
        branch_documents = reader.load_data(branch="main")
        index = VectorStoreIndex.from_documents(branch_documents)
        index.storage_context.persist()

    query_engine = index.as_query_engine()
    while True:
        conversation = {}
        question = input("\n Write your question or enter 'quit' to quit. \n\n")
        conversation[question] = ""

        if question == 'quit':
            break

        prompt = f"Respond to this question: {question} given the conversation history: {conversation} \n"
        response = query_engine.query(prompt)
        conversation[question] = response
        print(response)