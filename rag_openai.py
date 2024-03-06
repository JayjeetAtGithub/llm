import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
import PyPDF2


def read_pdf(pdf_path, outfile):
    pdfFileObj = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    for page in pdfReader.pages:
        outfile.write(page.extract_text())
    pdfFileObj.close()


if __name__ == "__main__":
    outfile = open('dataset_1/papers.txt', 'w')
    for file in os.listdir('./CXL_papers'):
        if file.endswith('.pdf'):
            pdf_path = os.path.join('./CXL_papers', file)
            read_pdf(pdf_path, outfile)
    outfile.close()

    PERSIST_DIR = "./papers_storage"
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader("dataset_1").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)


    query_engine = index.as_query_engine()
    while True:
        prompt = str(input("Enter your query: "))
        response = query_engine.query(prompt)
        print(response)
