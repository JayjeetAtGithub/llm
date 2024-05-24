

def generate_embedding(token):
    return [ord(char) for char in token]

if __name__ == "__main__":
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    for token in tokens:
        embedding = generate_embedding(token)

        index.add_item(embedding)

