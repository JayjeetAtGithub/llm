import nltk
import json
import multiprocessing as mp
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor
from wonderwords import RandomWord

load_dotenv(find_dotenv())
nltk.download('punkt')
client = OpenAI()


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.replace("\n", " ")
    return sentences


def read_txt_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_openai_embedding(sentence, model="text-embedding-3-small"):
   return client.embeddings.create(input = [sentence], model=model).data[0].embedding


def write_embeddings_to_file(embeddings_list):
    r = RandomWord()
    file_name = f"embeddings-{r.word(word_min_length=3, word_max_length=8)}.json"
    with open(file_name, "w") as file:
        file.write(json.dumps(embeddings_list))


def gen_embedding(sentence, idx):
    print(f"[INFO] Processing sentence with id: {idx}")
    sentence = sentence.strip()
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("\uffff", " ")
    vector_embedding = get_openai_embedding(sentence)
    return {
        "id": str(idx),
        "token": sentence,
        "embedding": vector_embedding
    }


if __name__ == "__main__":
    document = read_txt_file("papers_data/papers.txt")
    sentences = split_text_into_sentences(document)
    print(f"[INFO] Total sentences: {len(sentences)}")

    embeddings_list = list()
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures_to_openai = {executor.submit(gen_embedding, sentence, idx): idx for (idx, sentence) in enumerate(sentences[:2])}
        for future in futures_to_openai:
            embeddings_list.append(future.result())

    write_embeddings_to_file(embeddings_list)
    print("[INFO] Finished generating embeddings.")
