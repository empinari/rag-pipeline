from langchain_huggingface import HuggingFaceEmbeddings  # Updated this!
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient

DB_PATH = "./qdrant_local_v2"
COLLECTION_NAME = "advanced_rag_docs"

def run_rag_pipeline(query_text):
    # 1. Connect to our Qdrant vector store
    client = QdrantClient(path=DB_PATH)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # 2. Retrieve the top 2 or more by your preference for matching paragraphs (The Retrieval Step)
    print("Retrieving relevant context from database...")
    docs = vector_store.similarity_search(query=query_text, k=2)
    
  
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Build the prompt instructing the LLM to use that context
    prompt = f"""
    You are a helpful assistant. Answer the user's question using ONLY the provided context below. 
    If the context doesn't contain the answer, say "I cannot find that in the documents."
    
    Context:
    {context}
    
    Question: {query_text}
    
    Answer:
    """
    
    # 4. Pass everything to our local LLM (The Generation Step)
    print("Generating answer using local LLM...")
    llm = Ollama(model="llama3")
    response = llm.invoke(prompt)
    
    print("\n=== AI RESPONSE ===")
    print(response)
    print("===================\n")

if __name__ == "__main__":
    user_question = "What is the formal definition of machine learning according to Tom Mitchell?"
    run_rag_pipeline(user_question)