# evaluation.py

import os
from langchain_openai import ChatOpenAI
from vector_store import load_vector_store
from app import get_answer # Re-using the get_answer function from our app

# --- Setup ---
llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")

# Load your pre-built vector store
vector_store_path = "faiss_index_uploaded"
if os.path.exists(vector_store_path):
    db = load_vector_store(store_path=vector_store_path)
else:
    print("Vector store not found. Please run the app to create it first.")
    db = None

# --- Sample Evaluation Dataset ---
# In a real-world scenario, this would be a larger, more comprehensive dataset.
eval_dataset = [
    {
        "question": "What is the governing law of the agreement?",
        "ground_truth_answer": "The agreement is governed by the laws of the State of New York."
    },
    {
        "question": "What are the confidentiality obligations?",
        "ground_truth_answer": "Both parties must maintain the confidentiality of all proprietary information."
    }
]

def evaluate():
    if not db:
        return

    results = []

    for item in eval_dataset:
        question = item["question"]
        ground_truth = item["ground_truth_answer"]

        # --- Baseline (LLM only) ---
        baseline_answer = llm.invoke(question).content

        # --- RAG System ---
        rag_answer, _ = get_answer(db, question)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "baseline_answer": baseline_answer,
            "rag_answer": rag_answer
        })

    # --- Print Results for Manual Review ---
    for res in results:
        print("-" * 50)
        print(f"Question: {res['question']}")
        print(f"Ground Truth: {res['ground_truth']}")
        print(f"Baseline Answer: {res['baseline_answer']}")
        print(f"RAG Answer: {res['rag_answer']}")
        print("-" * 50)

    # Further steps would involve using metrics like ROUGE, BLEU, or semantic similarity
    # to programmatically compare the answers. For a legal context, manual review by a
    # domain expert is often the most reliable evaluation method.

if __name__ == '__main__':
    evaluate()