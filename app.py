# app.py

import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from document_processor import load_documents, chunk_documents

# --- Setup and Configuration ---
load_dotenv()
st.set_page_config(page_title="Legal Document Assistant", page_icon="⚖️", layout="wide")

# --- Constants ---
BASE_DOCS_FOLDER = "documents"
UPLOADED_DOCS_FOLDER = "uploaded_docs"
VECTOR_STORE_PATH = "faiss_combined_index"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RELEVANCE_THRESHOLD = 0.7  # Adjust this threshold based on testing

# --- LLM and Prompt Configuration ---
llm = ChatGroq(temperature=0.2, model_name="openai/gpt-oss-20b")
prompt_template = """
You are a helpful legal assistant. Your task is to answer questions based ONLY on the provided legal document excerpts.
If the provided context does not contain the answer to the question, you must state: "I'm sorry, but the answer to that question is not found in the provided documents."
Do not use any external knowledge.
Your answers should be clear, concise, and directly based on the information in the excerpts.

Context from documents:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- Helper Functions ---
@st.cache_resource
def get_embeddings_model():
    """Load the sentence transformer model."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

def create_or_load_vector_store():
    """Create a new vector store from base docs or load an existing one."""
    embeddings = get_embeddings_model()
    # If the vector store already exists, load it.
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # If not, create it from the base documents folder.
    st.info(f"Creating a new knowledge base from '{BASE_DOCS_FOLDER}'...")
    if not os.path.exists(BASE_DOCS_FOLDER) or not os.listdir(BASE_DOCS_FOLDER):
        os.makedirs(BASE_DOCS_FOLDER, exist_ok=True)
        # Return an empty store if base folder is empty
        return None 
    
    docs = load_documents(BASE_DOCS_FOLDER)
    if not docs: return None
    
    chunked_docs = chunk_documents(docs)
    db = FAISS.from_documents(chunked_docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    return db

def get_answer(db, query, k=4):
    """
    Performs a similarity search with relevance scores and generates an answer.
    """
    # Use similarity_search_with_relevance_scores to get scores
    retrieved_docs_with_scores = db.similarity_search_with_relevance_scores(query, k=k)

    # Filter documents based on relevance threshold
    relevant_docs = [doc for doc, score in retrieved_docs_with_scores if score < RELEVANCE_THRESHOLD]
    
    if not relevant_docs:
        return "I'm sorry, but I couldn't find any relevant information in the documents for your question.", []

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # answer = llm_chain.run({"context": context, "question": query})
    chain = prompt | llm
    answer_dict = chain.invoke({"context": context, "question": query})
    answer = answer_dict.content


    return answer, relevant_docs

def get_indexed_documents(db):
    """Retrieve the list of source documents from the vector store's metadata."""
    if not db:
        return []
    # The documents are stored in the docstore. We can access their metadata.
    all_docs = db.docstore._dict
    sources = set(doc.metadata.get('source', 'Unknown') for doc in all_docs.values())
    return sorted([os.path.basename(src) for src in sources])

# --- Main Application Logic ---
st.title("⚖️ RAG-based Legal Document Assistant")

# Initialize vector store
db = create_or_load_vector_store()

# --- Sidebar for Document Upload and Management ---
with st.sidebar:
    st.header("Document Management")
    
    # Create the folder for uploaded documents if it doesn't exist
    os.makedirs(UPLOADED_DOCS_FOLDER, exist_ok=True)

    uploaded_files = st.file_uploader(
        "Upload new documents to add to the knowledge base",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process and Add to Knowledge Base"):
            # Save uploaded files to the folder
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOADED_DOCS_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing documents and updating knowledge base..."):
                # Load and chunk the new documents
                new_docs = load_documents(UPLOADED_DOCS_FOLDER)
                if new_docs:
                    new_chunked_docs = chunk_documents(new_docs)
                    
                    # Add new documents to the existing vector store
                    if db:
                        db.add_documents(new_chunked_docs)
                    else: # If db was initially empty
                        embeddings = get_embeddings_model()
                        db = FAISS.from_documents(new_chunked_docs, embeddings)
                    
                    # Save the updated vector store and clear the upload folder
                    db.save_local(VECTOR_STORE_PATH)
                    shutil.rmtree(UPLOADED_DOCS_FOLDER)
                    os.makedirs(UPLOADED_DOCS_FOLDER) # Recreate empty folder
                    st.success("Knowledge base updated successfully!")
                    st.rerun() # Rerun to update the document list
                else:
                    st.warning("No new documents to process.")

    st.header("Indexed Documents")
    doc_list = get_indexed_documents(db)
    if doc_list:
        for doc_name in doc_list:
            st.info(doc_name)
    else:
        st.write("No documents have been indexed yet.")

# --- Main Interface for Q&A ---
st.header("Ask a Question")

if db is None:
    st.warning("The knowledge base is empty. Please add documents to begin.")
else:
    query = st.text_input("Enter your question about the indexed documents:")

    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching for answers..."):
                answer, retrieved_docs = get_answer(db, query)
                st.subheader("Answer:")
                st.markdown(answer)

                # Only show retrieved passages if they were found and deemed relevant
                if retrieved_docs:
                    st.subheader("Retrieved Passages and Citations:")
                    for i, doc in enumerate(retrieved_docs):
                        source_name = os.path.basename(doc.metadata.get('source', 'N/A'))
                        page_num = doc.metadata.get('page', 'N/A')
                        with st.expander(f"Source {i+1}: {source_name} - Page {page_num}"):
                            st.write(doc.page_content)