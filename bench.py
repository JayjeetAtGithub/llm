import os
import sys
import shutil
import platform
import argparse
import lancedb
import time
import uuid
import toml
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
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Constants
DEFAULT_CONFIG_FILE = "configs/default.toml"
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


def read_toml_file(file_path):
    with open(file_path, "r") as f:
        return toml.load(f)


def read_parquet_file(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    lst = df.values.tolist()
    return lst


def create_batches(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


def init_db_collection(config):
    if config["database"] == "chroma":
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection(config["table"])
    elif config["database"] == "lance":
        if os.path.exists("./lance_db"):
            shutil.rmtree("./lance_db")
        db = lancedb.connect("./lance_db")
        schema = pa.schema(
            [
                pa.field("embedding", pa.list_(pa.float32(), list_size=config["dimension"])),
                pa.field("token", pa.string()),
                pa.field("id", pa.int64()),
            ])
        collection = db.create_table(config["table"], schema=schema)
    elif config["database"] == "milvus":
        connections.connect("default", host="localhost", port="19530")
        if utility.has_collection(config["table"]):
            utility.drop_collection(config["table"])
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="token", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config["dimension"])
        ]
        schema = CollectionSchema(fields, config["table"])
        collection = Collection(config["table"], schema)
    elif config["database"] == "qdrant":
        if args.index:
            indexing_threshold = 20000
        else:
            indexing_threshold = 0
        collection = QdrantClient("localhost", port=6333)
        collection.delete_collection(collection_name=config["table"])
        collection.create_collection(
            collection_name=config["table"],
            vectors_config=VectorParams(size=config["dimension"], distance=Distance.DOT),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
    return collection


def init_client(config):
    if config["database"] == "qdrant":
        return QdrantClient("localhost", port=6333)


def run_query(config, client, vector):
    if config["database"] == "qdrant":
        results = client.search(
            collection_name=config["table"],
            query_vector=vector,
            with_vectors=False,
            with_payload=True,
            limit=config["top_k"],
        )
        print(f"Num results returned: {len(results)}")
        

def insert_into_collection_bulk(collection, batch, config):
    if config["database"] == "milvus":
        mini_batches = list(create_batches(batch, MILVUS_MAX_BATCH_SIZE))
        for b in mini_batches:
            s = time.time()
            collection.insert([
                # Add an Id to the row (probably uuid.uuid4() would be a good idea)
                [str(uuid.uuid4()) for _ in b],
                [row[config["payload_idx"]] for row in b],
                [list(row[config["embedding_idx"]]) for row in b],
            ])
            print(f"[INFO] Inserted batch of size {len(b)} in {time.time() - s} seconds")
    elif config["database"] == "chroma":
        collection.add(
            ids=[str(idx) for idx, _ in enumerate(batch)],
            documents=[row[config["payload_idx"]] for row in batch],
            embeddings=[list(row[config["embedding_idx"]]) for row in batch],
        )
    elif config["database"] == "lance":
        collection.add([
            {
                "id": idx, 
                "token": row[config["payload_idx"]], 
                "embedding": list(row[config["embedding_idx"]]), 
            } for idx, row in enumerate(batch)])
    elif config["database"] == "qdrant":
        mini_batches = list(create_batches(batch, QDRANT_MAX_BATCH_SIZE))
        for b in mini_batches:
            s = time.time()
            collection.upsert(
                collection_name=config["table"],
                wait=True,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        payload={"token": row[config["payload_idx"]]},
                        vector=list(row[config["embedding_idx"]]),
                    ) for row in b
                ],
            )
            print(f"[INFO] Inserted batch of size {len(b)} in {time.time() - s} seconds")


def get_collection_info(collection, config):
    if config["database"] == "milvus":
        collection.flush()
        print(collection.name)
        print(collection.schema)
        print(collection.num_entities)

    elif config["database"] == "chroma":
        print(collection.count())
    elif config["database"] == "lance":
        print(collection.schema)
        print(collection.count_rows())
    elif config["database"] == "qdrant":
        print("Collection info")
        print(collection.get_collection(collection_name=config["table"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--bench", type=str, default="qdrant-1M-1536", help="The benchmark setup to use")
    parser.add_argument("--debug", action="store_true", help="Whether to run the script in debug mode")
    parser.add_argument("--query", action="store_true", help="Whether to run a query on the collection")
    parser.add_argument("--index", action="store_true", help="Whether to index the collection")
    parser.add_argument("--ingest", action="store_true", help="Whether to ingest the embeddings into the collection")
    args = parser.parse_args()

    # Read the default config file
    config = read_toml_file(DEFAULT_CONFIG_FILE)
    config = {**config[args.bench]}
    print("[INFO] Running with config: ")
    print(config)

    # Ingest the dataset
    if args.ingest:
        total_rows_written = 0
        collection = init_db_collection(config)

        # Read the parquet files
        if args.debug:
            file_list = os.listdir(config["dataset"])[:2]
        else:
            file_list = os.listdir(config["dataset"])[config["train_start_idx"]:config["train_stop_idx"]]

        print(f"[INFO] Inserting {len(file_list)} files into collection {config['table']}")

        # Insert the embeddings into the collection
        for file in file_list:
            batch = read_parquet_file(os.path.join(config["dataset"], file))
            insert_into_collection_bulk(collection, batch, config)
            total_rows_written += len(batch)
            print(f"[INFO] Total rows written: {total_rows_written}")
        
        # Print out collection stats after insertion
        get_collection_info(collection, config)
    
    # Query the dataset
    if args.query:
        queries_ran = 0
        total_time_taken = 0
        client = init_client(config)

        file_list = os.listdir(config["dataset"])[config["query_start_idx"]:config["query_stop_idx"]]
        print(f"[INFO] Running queries from {len(file_list)} files")

        for file in file_list:
            for row in read_parquet_file(os.path.join(config["dataset"], file)):
                vector = row[config["embedding_idx"]]
                print(f"[INFO] Running query #{queries_ran} for vector: [{vector[0]}, {vector[1]}, {vector[2]}, ...]")
                s = time.time()
                run_query(config, client, vector)
                total_time_taken += time.time() - s
                queries_ran += 1
                if queries_ran >= config["queries_to_run"]:
                    break
    
        print(f"Total time for {config['queries_to_run']} queries: {total_time_taken}")
    print("[INFO] Done!")
