import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_PATH = "docs"
CHROMA_PATH = "data/chroma"

def load_documents():
    documents = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_PATH, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    print("\nLoading embedding model (first time may take 1-2 mins)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("Embedding and storing chunks in ChromaDB...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"\nDone! Vectorstore saved to: {CHROMA_PATH}")
    return vectorstore

if __name__ == "__main__":
    print("=== LEO — Ingestion Pipeline ===\n")
    documents = load_documents()
    chunks = split_documents(documents)
    create_vectorstore(chunks)
    print("\nIngestion complete. LEO is ready to answer questions!")
