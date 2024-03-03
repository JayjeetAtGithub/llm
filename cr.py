import os
import shutil
import json
import platform
import argparse
import lancedb
import pyarrow as pa
from dotenv import load_dotenv, find_dotenv
from pyinstrument import Profiler

# Import ChromaDB properly
if platform.system() == "Linux":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import chromadb

# Load the environment variables
load_dotenv(find_dotenv())


def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.loads(file.read())
    

def insert_into_collection(collection, embedding, db):
    if db == "chroma":
        collection.add(
            documents=[embedding["token"]],
            ids=[str(embedding["id"])],
            metadatas={"id": str(embedding["id"])},
            embeddings=[embedding["embedding"]],
        )
    elif db == "lancedb":
        collection.add([
            {
                "embedding": embedding["embedding"], 
                "document": embedding["token"], 
                "id": embedding["id"], 
                "metadata": {"id": str(embedding["id"])}
            }])


if __name__ == "__main__":
    # The vector database to use
    parser = argparse.ArgumentParser() 
    parser.add_argument("--db", type=str, default="chroma", help="The vector database to use (lancedb/chromadb)")   
    args = parser.parse_args()

    profiler = Profiler()

    if args.db == "chroma":
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection("embeddings_table")
    elif args.db == "lance":
        if os.path.exists("./lance_db"):
            shutil.rmtree("./lance_db")
        db = lancedb.connect("./lance_db")
        schema = pa.schema(
            [
                pa.field("embedding", pa.list_(pa.float32(), list_size=1578)),
                pa.field("document", pa.string()),
                pa.field("id", pa.string()),
                pa.field("metadata", pa.struct([("id", pa.string())]))
            ])
        tbl = db.create_table("embeddings_table", schema=schema)


    embeddings_list = read_json_file("embeddings.json")
    print(f"[INFO] Total embeddings read: {len(embeddings_list)}")

    profiler.start()
    for embedding in embeddings_list:
        insert_into_collection(collection, embedding, args.db)
        print(f"[INFO] Added {embedding['id']} to the collection.")
    profiler.stop()

    print(collection.count())
    profiler.open_in_browser()
