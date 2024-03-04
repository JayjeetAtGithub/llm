import os
import sys
import shutil
import json
import platform
import argparse
import lancedb
import pyarrow as pa
from dotenv import load_dotenv, find_dotenv
from pyinstrument import Profiler
from deeplake.core.vectorstore import VectorStore
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct


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
    

def init_db_collection(db):
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
                pa.field("embedding", pa.list_(pa.float32(), list_size=1536)),
                pa.field("token", pa.string()),
                pa.field("id", pa.string()),
            ])
        collection = db.create_table("embeddings_table", schema=schema)
    elif args.db == "deeplake":
        print("DeepLake implementation not available yet.")
        sys.exit(0)
        if os.path.exists("./deeplake_db"):
            shutil.rmtree("./deeplake_db")
        collection = VectorStore(path="./deeplake_db")
    elif args.db == "milvus":
        connections.connect("default", host="localhost", port="19530")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="token", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        schema = CollectionSchema(fields, "embeddings_table")
        collection = Collection("embeddings_table", schema)
    elif args.db == "qdrant":
        if os.path.exists("./qdrant_db"):
            shutil.rmtree("./qdrant_db")
        collection = QdrantClient(url="http://localhost:6333")
        collection.create_collection(
            collection_name="embeddings_table",
            vectors_config=VectorParams(size=1536, distance=Distance.DOT),
        )
    return collection


def insert_into_collection_bulk(collection, embeddings, db):
    if db == "milvus":
        collection.insert([
            [embedding["id"] for embedding in embeddings],
            [embedding["token"] for embedding in embeddings],
            [embedding["embedding"] for embedding in embeddings]
        ])
    elif db == "chroma":
        collection.add(
            documents=[embedding["token"] for embedding in embeddings],
            ids=[str(embedding["id"]) for embedding in embeddings],
            embeddings=[embedding["embedding"] for embedding in embeddings],
        )
    elif db == "lance":
        collection.add([
            {
                "embedding": embedding["embedding"], 
                "token": embedding["token"], 
                "id": str(embedding["id"]), 
            } for embedding in embeddings])
    elif db == "qdrant":
        collection.upsert(
            collection_name="embeddings_table",
            wait=True,
            points=[
                PointStruct(
                    id=embedding["id"], 
                    vector=embedding["embedding"], 
                    payload={"token": embedding["token"]},
                ) for embedding in embeddings
            ],
        )
        

def insert_into_collection(collection, embedding, db):
    if db == "chroma":
        collection.add(
            documents=[embedding["token"]],
            ids=[str(embedding["id"])],
            embeddings=[embedding["embedding"]],
        )
    elif db == "lance":
        collection.add([
            {
                "embedding": embedding["embedding"], 
                "token": embedding["token"], 
                "id": str(embedding["id"]), 
            }])
    elif db == "deeplake":
        collection.add(
            text = embedding["token"],
            embedding = embedding["embedding"],
            metadata = {"id": str(embedding["id"])},
        )
    elif db == "milvus":
        collection.insert([
            [embedding["id"]],
            [embedding["token"]],
            [embedding["embedding"]]
        ])
    elif db == "qdrant":
        collection.upsert(
            collection_name="embeddings_table",
            wait=True,
            points=[
                PointStruct(
                    id=embedding["id"], 
                    vector=embedding["embedding"], 
                    payload={"token": embedding["token"]},
                )
            ],
        )


def get_collection_info(collection, db):
    if db == "milvus":
        collection.flush()
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(
            field_name="embedding", 
            index_params=index_params
        )

        print(collection.name)
        print(collection.schema)
        print(collection.num_entities)

        collection.load(replica_number=1)
    elif db == "chroma":
        print(collection.count())
    elif db == "lance":
        print(collection.schema)
        print(collection.count_rows())


if __name__ == "__main__":
    # The vector database to use
    parser = argparse.ArgumentParser() 
    parser.add_argument("--db", type=str, default="milvus", help="The vector database to use (lancedb/chromadb/deeplake/milvus)")
    parser.add_argument("--embeddings", type=str, default="embeddings.json", help="The embeddings file to read from")
    parser.add_argument("--bulk", action="store_true", help="Whether to bulk insert the embeddings")   
    args = parser.parse_args()

    # Instantiate the profiler
    profiler = Profiler()

    # Read out the em,beddings from the JSON file into memory
    embeddings_list = read_json_file(args.embeddings)
    print(f"[INFO] Total embeddings read: {len(embeddings_list)}")

    # Initialize the collection
    collection = init_db_collection(args.db)

    profiler.start()
    if args.bulk:
        insert_into_collection_bulk(collection, embeddings_list, args.db)
        print(f"[INFO] Bulk added {len(embeddings_list)} embeddings to the {args.db} collection")
    else:
        for embedding in embeddings_list:
            insert_into_collection(collection, embedding, args.db)
            print(f"[INFO] Added embedding {embedding['id']} to the {args.db} collection")
    profiler.stop()

    # Print out collection stats
    get_collection_info(collection, args.db)

    # Open the pyinstrument profile in the browser
    profiler.open_in_browser()
