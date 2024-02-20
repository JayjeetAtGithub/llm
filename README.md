# rag-app

A RAG (Retrieval Augmented Generation) application built using Llama-Index to browse research papers. We use the LLama 7B Chat model quantized into 3 bits from HuggingFace and use LanceDB as our vector database.

## Running Instructions

1. Clone the repository.
```bash
git clone https://github.com/JayjeetAtGithub/rag-app
cd rag-app/
```

2. Install dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r req.txt
```

3. Read PDF files into text documents.
```bash
python3 populate_data.py
```

4. Run the RAG application.
```bash
python3 rag_hf.py
```
