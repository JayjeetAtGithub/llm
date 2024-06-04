import os
import time
import argparse

import psycopg2
import pyarrow.parquet as pq
from pgvector.psycopg2 import register_vector


def read_parquet_file(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    lst = df.values.tolist()
    return lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--query", action="store_true", help="Whether to run a query on the collection")
    parser.add_argument("--index", action="store_true", help="Whether to create an index on the collection")
    parser.add_argument("--ingest", action="store_true", help="Whether to ingest the embeddings into the collection")
    args = parser.parse_args()

    conn = psycopg2.connect("postgresql://noobjc@localhost:5432/vectordb", auto_commit=True)
    print("Connected to PostgreSQL: ", conn)

    if args.ingest:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
        print("Initiated extension: pg_vector")

        conn.execute('DROP TABLE IF EXISTS embeddings_table')
        conn.execute('CREATE TABLE embeddings_table (id bigserial PRIMARY KEY, content text, embedding vector(1536))')
        print("Created table: embeddings_table")

        file_list = os.listdir("dbpedia-entities-openai-1M/data")
        row_idx = 0
        for file in file_list:
            batch = read_parquet_file(os.path.join("dbpedia-entities-openai-1M/data", file))
            for row in batch:
                conn.execute('INSERT INTO embeddings_table (content, embedding) VALUES (%s, %s)', (row[2], row[3]))
                print(f"Inserted row {row_idx} into pg_vector")
                row_idx += 1

        print(f"Inserted {row_idx} rows into pg_vector")
    
    if args.index:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
        print("Initiated extension: pg_vector")

        conn.execute('SET max_parallel_maintenance_workers = 40;')
        conn.execute('SET max_parallel_workers = 40;')
        conn.execute('SET maintenance_work_mem = "64GB";')
        conn.execute('CREATE INDEX ON embeddings_table USING hnsw (embedding vector_l2_ops) WITH (m = 32, ef_construction = 32);')
        print("Created index on embeddings_table using HNSW algorithm")

    if args.query:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
        print("Initiated extension: pg_vector")

        file_list = os.listdir("dbpedia-entities-openai-1M/data")[5:7]

        query_idx = 0
        for file in file_list:
            batch = read_parquet_file(os.path.join("dbpedia-entities-openai-1M/data", file))
            for row in batch:
                res = conn.execute(f"BEGIN; SET LOCAL hnsw.ef_search = 100; SELECT * FROM embeddings_table ORDER BY embedding <-> '{row[3].tolist()}' LIMIT 100; COMMIT;").fetchall()
                print(f"Ran query {query_idx} on pg_vector")
                query_idx += 1
