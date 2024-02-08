import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage, StorageContext
from llama_index.llms import OpenAI
from pathlib import Path
import PyPDF2
from llama_index import download_loader


def read_pdf(pdf_path, outfile):
    pdfFileObj = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    for page in pdfReader.pages:
        outfile.write(page.extract_text())
    pdfFileObj.close()


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    outfile = open('papers_data/papers.txt', 'w')
    for file in os.listdir('./CXL_papers'):
        if file.endswith('.pdf'):
            pdf_path = os.path.join('./CXL_papers', file)
            read_pdf(pdf_path, outfile)
    outfile.close()

    PERSIST_DIR = "./papers_storage"
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        service_context = ServiceContext.from_defaults(llm=llm)
        documents = SimpleDirectoryReader("papers_data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)


    query_engine = index.as_query_engine(similarity_top_k=5)
    while True:
        prompt = str(input("Enter your query: "))
        response = query_engine.query(prompt)
        print(response)
