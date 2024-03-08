import os
import sys
import shutil
import platform
import argparse
import lancedb
import time
import uuid
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv, find_dotenv
from pymilvus import (
    utility,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

# For the current dataset,
MILVUS_MAX_BATCH_SIZE = 10000
QDRANT_MAX_BATCH_SIZE = 1000

# Import ChromaDB properly
if platform.system() == "Linux":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Load the environment variables
load_dotenv(find_dotenv())


def read_parquet_file(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    lst = df.values.tolist()
    return lst


def create_batches(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
    

def init_db_collection(args):
    if args.db == "chroma":
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection(args.tbl)
    elif args.db == "lance":
        if os.path.exists("./lance_db"):
            shutil.rmtree("./lance_db")
        db = lancedb.connect("./lance_db")
        schema = pa.schema(
            [
                pa.field("embedding", pa.list_(pa.float32(), list_size=args.dim)),
                pa.field("token", pa.string()),
                pa.field("id", pa.int64()),
            ])
        collection = db.create_table(args.tbl, schema=schema)
    elif args.db == "milvus":
        connections.connect("default", host="localhost", port="19530")
        if utility.has_collection(args.tbl):
            utility.drop_collection(args.tbl)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="token", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dim)
        ]
        schema = CollectionSchema(fields, args.tbl)
        collection = Collection(args.tbl, schema)
    elif args.db == "qdrant":
        collection = QdrantClient("localhost", port=6333)
        collection.delete_collection(collection_name="embeddings_table")
        collection.create_collection(
            collection_name=args.tbl,
            vectors_config=VectorParams(size=args.dim, distance=Distance.DOT),
        )
    return collection


def insert_into_collection_bulk(collection, batch, args):
    if args.db == "milvus":
        mini_batches = list(create_batches(batch, MILVUS_MAX_BATCH_SIZE))
        for b in mini_batches:
            s = time.time()
            collection.insert([
                [row[2] for row in b],
                [list(row[3]) for row in b],
            ])
            print(f"[INFO] Inserted batch of size {len(b)} in {time.time() - s} seconds")
    elif args.db == "chroma":
        collection.add(
            ids=[str(idx) for idx, _ in enumerate(batch)],
            documents=[row[2] for row in batch],
            embeddings=[list(row[3]) for row in batch],
        )
    elif args.db == "lance":
        collection.add([
            {
                "id": idx, 
                "token": row[2], 
                "embedding": list(row[3]), 
            } for idx, row in enumerate(batch)])
    elif args.db == "qdrant":
        mini_batches = list(create_batches(batch, QDRANT_MAX_BATCH_SIZE))
        for b in mini_batches:
            s = time.time()
            collection.upsert(
                collection_name=args.tbl,
                wait=True,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        payload={"token": row[2]},
                        vector=list(row[3]),
                    ) for row in b
                ],
            )
            print(f"[INFO] Inserted batch of size {len(b)} in {time.time() - s} seconds")


def get_collection_info(collection, args):
    if args.db == "milvus":
        collection.flush()
        print(collection.name)
        print(collection.schema)
        print(collection.num_entities)
        
        # Optionally, load the milvus collection on
        # demand from the user
        if args.load_milvus:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="embedding", 
                index_params=index_params
            )
            collection.load(replica_number=1)

    elif args.db == "chroma":
        print(collection.count())
    elif args.db == "lance":
        print(collection.schema)
        print(collection.count_rows())
    elif args.db == "qdrant":
        print("Collection info")
        print(collection.get_collection(collection_name=args.tbl))


if __name__ == "__main__":
    # The vector database to use
    parser = argparse.ArgumentParser() 
    parser.add_argument("--db", type=str, default="milvus", help="The vector database to use (lance/chroma/milvus/qdrant)")
    parser.add_argument("--ds", type=str, default="embeddings_dataset", help="The embeddings directory to read from")
    parser.add_argument("--dim", type=int, default=1536, help="The dimension of the vector embeddings")
    parser.add_argument("--tbl", type=str, default="embeddings_table", help="The table name to use in the database")
    parser.add_argument("--load-milvus", action="store_true", help="Whether to load the Milvus collection")
    args = parser.parse_args()

    # Initialize the collection
    collection = init_db_collection(args)

    for file in os.listdir(args.ds)[:4]:
        batch = read_parquet_file(os.path.join(args.ds, file))
        insert_into_collection_bulk(collection, batch, args)
        print(f"[INFO] Bulk added {len(batch)} embeddings to the {args.db} collection")

    # Print out collection stats
    get_collection_info(collection, args)
