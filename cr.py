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

load_dotenv(find_dotenv())

def read_json_file(file_path):
    with open(file_path, "r") as file:
        return file.read()
    


if __name__ == "__main__":
    profiler = Profiler()

    # Remove previous instances
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Instantiate the ChromaDB client and collection
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    profiler.start()
    # Generate embeddings for each sentence
    for idx, sentence in enumerate(sentences[:2]):
        chroma_collection.add(
            documents=[sentence],
            ids=[f"id{idx}"],
            metadatas={"id": idx},
            embeddings=embeddings_list[idx],
        )
        print(f"Added {idx}")
    profiler.stop()

    print(chroma_collection.count())
    profiler.print()
