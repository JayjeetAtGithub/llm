import nltk
import json
import multiprocessing as mp
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor
from wonderwords import RandomWord
from sentence_transformers import SentenceTransformer

load_dotenv(find_dotenv())
nltk.download('punkt')


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.replace("\n", " ")
    return sentences


def read_txt_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_embedding(sentence, model):
    embedding = model.encode([sentence])
    return embedding[0].tolist()


def write_embeddings_to_file(embeddings_list):
    r = RandomWord()
    file_name = f"/mnt/workspace/embeddings-{r.word(word_min_length=3, word_max_length=8)}.json"
    with open(file_name, "w") as file:
        file.write(json.dumps(embeddings_list))


def gen_embedding(sentence, idx, model):
    print(f"[INFO] Processing sentence with id: {idx}")
    sentence = sentence.strip()
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("\uffff", " ")
    vector_embedding = get_embedding(sentence, model)
    return {
        "id": str(idx),
        "token": sentence,
        "embedding": vector_embedding
    }


if __name__ == "__main__":
    document = read_txt_file("/mnt/workspace/kernel.txt")
    sentences = split_text_into_sentences(document)
    print(f"[INFO] Total sentences: {len(sentences)}")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings_list = list()
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures_to_openai = {executor.submit(gen_embedding, sentence, idx, model): idx for (idx, sentence) in enumerate(sentences[:2])}
        for future in futures_to_openai:
            embeddings_list.append(future.result())

    write_embeddings_to_file(embeddings_list)
    print("[INFO] Finished generating embeddings.")
