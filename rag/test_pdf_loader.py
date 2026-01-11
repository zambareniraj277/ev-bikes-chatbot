from rag.pdf_loader import load_pdfs_from_folder, chunk_texts

folder_path = "D:/datachamps asgg/EV Chatbot/data/Annual Reports-20250804T101032Z-1-001/Annual Reports"
texts = load_pdfs_from_folder(folder_path)
chunks = chunk_texts(texts)

for i in range(min(5, len(chunks))):
    print(f"\n🔹 Chunk {i+1}:\n{chunks[i][:300]}...\n{'-'*60}")
