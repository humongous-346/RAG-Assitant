# ⚖️ RAG-based Legal Document Assistant

A powerful Retrieval-Augmented Generation (RAG) application designed specifically for legal document analysis and question-answering. This application allows users to upload legal documents, create a searchable knowledge base, and get precise answers based on the document content.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kf9ldwzz8xgag2dbircc2b.streamlit.app/)

> **Note**: To deploy your own instance, follow the [deployment guide](#deployment) below.

## 🌟 Features

- **Document Upload & Processing**: Support for PDF and DOCX file formats
- **Intelligent Text Chunking**: Optimized document splitting for better retrieval
- **Vector-based Search**: Uses FAISS for efficient similarity search
- **Relevance Filtering**: Smart filtering to ensure only relevant content is used
- **Interactive Web Interface**: Clean Streamlit-based UI for easy interaction
- **Citation Support**: Shows source documents and page numbers for answers
- **Knowledge Base Management**: Add new documents to existing knowledge base
- **Legal-focused Prompting**: Specialized prompts for legal document analysis

## 🏗️ Architecture

The application follows a modular RAG architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Text Processing │───▶│ Vector Database │
│  (PDF/DOCX)     │    │   & Chunking     │    │     (FAISS)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   User Query    │───▶│  Similarity      │◀────────────┘
└─────────────────┘    │     Search       │
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐
│  Final Answer   │◀───│    LLM Chain     │
│  with Sources   │    │  (ChatGroq)      │
└─────────────────┘    └──────────────────┘
```

## 📁 Project Structure

```
RAG-Assistant/
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document loading and chunking logic
├── vector_store.py        # Vector database creation and management
├── evaluation.py          # Model evaluation utilities
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── documents/             # Base documents folder
├── uploaded_docs/         # Temporary folder for uploaded files
├── faiss_combined_index/  # FAISS vector store
└── screenshots/           # Application screenshots
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/humongous-346/RAG-Assitant
   cd RAG-Assistant
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

## 📚 Usage Guide

### Adding Documents

1. **Upload Files**: Use the sidebar to upload PDF or DOCX files
2. **Process Documents**: Click "Process and Add to Knowledge Base"
3. **Verify Addition**: Check the "Indexed Documents" section to confirm

### Asking Questions

1. **Enter Query**: Type your question in the main text input
2. **Get Answer**: Click "Get Answer" to retrieve information
3. **Review Sources**: Expand the source citations to see original passages


### Embedding Models

The application uses Sentence Transformers by default. Supported models include:
- `all-MiniLM-L6-v2` (default) - Fast and efficient
- `all-mpnet-base-v2` - Higher quality, slower
- `paraphrase-MiniLM-L6-v2` - Good for paraphrase detection

## 🌐 Deployment

### Deploy to Streamlit Community Cloud (Recommended)

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub account** and select this repository

4. **Set the main file path** to `app.py`

5. **Add secrets** in the Streamlit dashboard:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

6. **Deploy** - Your app will be available at `https://your-app-name.streamlit.app`


## 🔧 Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and document processing
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **ChatGroq**: Language model API
- **PyPDF**: PDF document processing
- **python-docx/docx2txt**: Word document processing

### Vector Store

- **Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Dimension**: 384 (for all-MiniLM-L6-v2)
- **Distance Metric**: L2 (Euclidean distance)
- **Index Type**: Flat (exact search)

## 🎯 Evaluation

The project includes evaluation capabilities in `evaluation.py` for:
- Retrieval accuracy assessment
- Answer quality evaluation
- Performance benchmarking

