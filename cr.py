import os
import nltk
import shutil
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pyinstrument import Profiler

# Initializations
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

load_dotenv(find_dotenv())
nltk.download('punkt')
client = OpenAI()

def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def read_txt_file(file_path):
    with open(file_path, "r") as file:
        return file.read()
    

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


if __name__ == "__main__":
    profiler = Profiler()

    # Remove previous instances
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Instantiate the ChromaDB client and collection
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    # Read data and chunk them into sentences
    document = read_txt_file("papers_data/papers.txt")
    sentences = split_text_into_sentences(document)

    embeddings_list = list()
    for sentence in sentences:
        embeddings_list.append(get_embedding(sentence))

    profiler.start()
    # Generate embeddings for each sentence
    for idx, sentence in enumerate(sentences[:100]):
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
