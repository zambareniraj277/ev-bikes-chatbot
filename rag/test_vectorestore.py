from pdf_loader import load_pdfs_from_folder
from vectorstore import get_vectorstore

folder_path = "D:/datachamps asgg/EV Chatbot/data/Annual Reports-20250804T101032Z-1-001/Annual Reports"
texts = load_pdfs_from_folder(folder_path)

# If you're using sentence transformers
vectorstore = get_vectorstore(texts)

print("✅ Vectorstore created successfully.")
