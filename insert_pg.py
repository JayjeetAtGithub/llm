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

    conn = psycopg.connect(dbname='vectordb', autocommit=True)

    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)

    conn.execute('DROP TABLE IF EXISTS embeddings_table')
    conn.execute('CREATE TABLE embeddings_table (id bigserial PRIMARY KEY, content text, embedding vector(1536))')

    file_list = os.listdir("dbpedia-entities-openai-1M/data")
    for file in file_list:
        batch = read_parquet_file(os.path.join("dbpedia-entities-openai-1M/data", file))
        for row in batch:
            conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (row[2], row[3]))        
            print(f"Inserted row into pg_vector")
