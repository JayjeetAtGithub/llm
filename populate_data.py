from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", cache_dir="hf_dataset")    
    print("Data downloaded successfully")
