import os
import PyPDF2
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex


def read_pdf(pdf_path, outfile):
    pdfFileObj = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    for page in pdfReader.pages:
        outfile.write(page.extract_text())
    pdfFileObj.close()


if __name__ == "__main__":
    os.environ["HF_HOME"] = "./cache"

    outfile = open('papers_data/papers.txt', 'w')
    for file in os.listdir('./CXL_papers'):
        if file.endswith('.pdf'):
            pdf_path = os.path.join('./CXL_papers', file)
            read_pdf(pdf_path, outfile)
    outfile.close()

    llm = "local"
    embed_model = "local"

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    documents = SimpleDirectoryReader("papers_data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine(similarity_top_k=5)
    while True:
        prompt = str(input("Enter your query: "))
        response = query_engine.query(prompt)
        print(response)
