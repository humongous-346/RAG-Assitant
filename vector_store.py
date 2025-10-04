# vector_store.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from document_processor import load_documents, chunk_documents

# It's recommended to set your OpenAI API key as an environment variable
# For example: export OPENAI_API_KEY='your_key_here'
# If you can't, you can pass it directly: OpenAIEmbeddings(openai_api_key="your_key")

def create_vector_store(chunks, embedding_model_name='all-MiniLM-L6-v2', store_path="faiss_index"):
    """
    Generates embeddings and creates a FAISS vector store.
    """
    # Using Sentence-BERT embeddings
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Uncomment the line below to use OpenAI embeddings instead
    # embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(store_path)
    print(f"Vector store created and saved at {store_path}")
    return db

def load_vector_store(store_path="faiss_index", embedding_model_name='all-MiniLM-L6-v2'):
    """
    Loads an existing FAISS vector store.
    """
    # Using Sentence-BERT embeddings
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Uncomment the line below to use OpenAI embeddings instead
    # embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    return db


if __name__ == '__main__':
    documents_folder = 'documents'
    if not os.path.exists(documents_folder) or not os.listdir(documents_folder):
         os.makedirs(documents_folder, exist_ok=True)
         print(f"The '{documents_folder}' folder is empty. Please add legal documents to it and run this script again.")
    else:
        # 1. Load and process documents
        docs = load_documents(documents_folder)
        chunked_docs = chunk_documents(docs)
        print(f"Processed {len(chunked_docs)} text chunks.")

        # 2. Create and save the vector store
        vector_store = create_vector_store(chunked_docs)

        # 3. Example of loading the vector store and performing a search
        print("\nLoading the vector store for a test search...")
        db = load_vector_store()
        query = "What are the key obligations of the tenant?"
        search_results = db.similarity_search(query, k=2) # k is the number of results

        print(f"\nTop 2 search results for the query: '{query}'")
        for i, result in enumerate(search_results):
            print(f"\n--- Result {i+1} ---")
            print("Content:", result.page_content)
            print("Source:", result.metadata.get('source', 'N/A'))
            print("Page:", result.metadata.get('page', 'N/A'))