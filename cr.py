import os
import shutil
import json
from dotenv import load_dotenv, find_dotenv
from pyinstrument import Profiler

# Import ChromaDB properly
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Load the environment variables
load_dotenv(find_dotenv())


def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.loads(file.read())


if __name__ == "__main__":
    profiler = Profiler()

    # Remove previous instances and instantiate the ChromaDB client and collection
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    # Read the embeddings from the file
    embeddings_list = read_json_file("embeddings.json")
    print(f"[INFO] Total embeddings read: {len(embeddings_list)}")

    profiler.start()
    # Generate embeddings for each sentence
    for embedding in embeddings_list:
        chroma_collection.add(
            documents=[embedding["token"]],
            ids=[str(embedding["id"])],
            metadatas={"id": str(embedding["id"])},
            embeddings=[embedding["embedding"]],
        )
        print(f"[INFO] Added {embedding['id']} to the collection.")
    profiler.stop()

    print(chroma_collection.count())
    profiler.print()
