import os

import psycopg
import pyarrow.parquet as pq
from pgvector.psycopg import register_vector


def read_parquet_file(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    lst = df.values.tolist()
    return lst

if __name__ == "__main__":

    conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/vectordb", autocommit=True)
    print("Connected to PostgreSQL: ", conn)

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
            conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (row[2], row[3]))        
            print(f"Inserted row {row_idx} into pg_vector")
            row_idx += 1

    print(f"Inserted {row_idx} rows into pg_vector")
