import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configuration variables
DATA_DIR = "./data"
DB_PATH = "./qdrant_local_v2"
COLLECTION_NAME = "advanced_rag_docs"

def load_and_chunk_documents():
    """Reads raw text files and splits them into clean chunks."""
    print("Reading documents from data folder...")
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(f"Please put a text file inside the '{DATA_DIR}' folder first.")

    raw_text = ""
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                raw_text += f.read() + "\n\n"

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    
    chunks = text_splitter.create_documents(
        texts=[raw_text], 
        metadatas=[{"source": "portfolio_dataset"}]
    )
    print(f"Successfully split text into {len(chunks)} chunks.")
    return chunks

def initialize_vector_store(documents):
    """Initializes local Qdrant database and indexes the text chunks."""
    print("Initializing local embedding engine...")
    
    # Downloads a free local embedding model from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build or connect to a local on-disk Qdrant instance
    client = QdrantClient(path=DB_PATH)
    
    # Set up our database table (collection)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
    print("Transforming text into vectors and saving to Qdrant...")
    
    # 1. Instantiate the vector store directly with the open client
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # 2. Add the documents to the initialized store
    vector_store.add_documents(documents=documents)
    
    print(f"Success! Your vectors are saved locally inside: '{DB_PATH}'")

if __name__ == "__main__":
    try:
        chunks = load_and_chunk_documents()
        initialize_vector_store(chunks)
    except Exception as e:
        print(f"Error occurred: {e}")