from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def get_vectorstore(text_chunks: list, persist_directory: str = "data/output/chroma") -> Chroma:
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Smaller model, faster and works out of the box
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    print("✅ Embedding + Vector store created.")
    return vectorstore
