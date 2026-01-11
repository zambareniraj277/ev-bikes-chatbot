import os
import fitz  # PyMuPDF
from typing import List

def load_pdfs_from_folder(folder_path: str) -> List[str]:
    all_texts = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                print(f"📄 Loading: {file_path}")
                text = extract_text_from_pdf(file_path)
                all_texts.append(text)

    return all_texts

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts: List[str], chunk_size=1000, chunk_overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for text in texts:
        chunks.extend(splitter.split_text(text))

    print(f"✅ Total Chunks Created: {len(chunks)}")
    return chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks
