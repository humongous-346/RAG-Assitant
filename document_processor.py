# document_processor.py

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def load_documents(folder_path):
    """
    Loads documents from the specified folder.
    Supports PDF and DOCX files.
    """
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    return documents

def clean_text(text):
    """
    Applies basic cleaning to the text.
    Removes extra whitespace and can be extended to handle OCR noise.
    """
    # Remove excessive newlines and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Add more specific cleaning rules here if needed
    return text

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Clean the content of each chunk
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    return chunks

if __name__ == '__main__':
    # Create a 'documents' folder in your project directory and add some sample PDF and DOCX files.
    documents_folder = 'documents'
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)
        print(f"Created a '{documents_folder}' folder. Please add your legal documents there.")
    else:
        # Load the documents
        docs = load_documents(documents_folder)
        print(f"Loaded {len(docs)} documents.")

        # Chunk the documents
        chunked_docs = chunk_documents(docs)
        print(f"Split documents into {len(chunked_docs)} chunks.")
        # You can print a sample chunk to see the result
        if chunked_docs:
            print("\nSample Chunk:")
            print(chunked_docs[0].page_content)
            print("\nMetadata:")
            print(chunked_docs[0].metadata)