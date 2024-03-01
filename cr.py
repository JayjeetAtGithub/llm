import os
import chromadb
import nltk
import shutil
from sentence_transformers import SentenceTransformer

os.environ["OPENAI_API_KEY"] = "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"
nltk.download('punkt')


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


if __name__ == "__main__":
    from llama_index.core import SimpleDirectoryReader
    from llama_index.embeddings.openai import OpenAIEmbedding

    # Remove previous instances
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Instantiate the ChromaDB client and collection
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    # Read data and chunk them into sentences
    documents = SimpleDirectoryReader("papers_data").load_data()
    sentences = split_text_into_sentences(documents[0].text)

    # Generate embeddings for each sentence
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    for idx, sentence in enumerate(sentences[:100]):
        embeddings = embed_model.get_text_embedding(sentence)
        chroma_collection.add(
            documents=[sentence],
            ids=[f"id{idx}"],
            metadatas={"id": idx},
            embeddings=embeddings,
        )
        print(f"Added {idx}")

    print(chroma_collection.peek())
    print(chroma_collection.count())
