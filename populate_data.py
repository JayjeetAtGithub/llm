import os
import PyPDF2

def read_pdf(pdf_path, outfile):
    pdfFileObj = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    for page in pdfReader.pages:
        outfile.write(page.extract_text())
    pdfFileObj.close()

if __name__ == '__main__':
    outfile = open('dataset_1/papers.txt', 'w')
    for file in os.listdir('./CXL_papers'):
        print("Reading file: ", file)
        if file.endswith('.pdf'):
            pdf_path = os.path.join('./CXL_papers', file)
            read_pdf(pdf_path, outfile)
    outfile.close()
