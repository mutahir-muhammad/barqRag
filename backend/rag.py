from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from spacy.lang.en import English
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure CPU usage
device = "cpu"

# Load models
logging.info("Loading sentence transformer model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
logging.info("Loading LLM model...")
llm_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={"": "cpu"}
)

# Set up upload directory
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# NLP Pipeline for sentence splitting
nlp = English()
nlp.add_pipe("sentencizer")

def read_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text").replace("\n", " ").strip() for page in tqdm(doc, desc="Extracting PDF text")]
    if not any(pages):
        raise ValueError("No text extracted from PDF. Check if the document contains selectable text.")
    return pages

def split_sentences(text):
    """Splits text into sentences using spaCy."""
    return [str(sent).strip() for sent in nlp(text).sents if str(sent).strip()]

def chunk_sentences(sentences, chunk_size=10):
    """Splits a list of sentences into chunks of `chunk_size` sentences."""
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles PDF file upload and processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)
    
    # Extract text from PDF
    pages_text = read_pdf(pdf_path)
    
    # Process text into chunks
    pages_sentences = [split_sentences(page) for page in tqdm(pages_text, desc="Splitting into sentences")]
    pages_chunks = [chunk_sentences(sentences) for sentences in pages_sentences]
    flat_chunks = [(page_no + 1, chunk) for page_no, chunks in enumerate(pages_chunks) for chunk in chunks]
    
    # Convert to DataFrame
    chunks_df = pd.DataFrame(flat_chunks, columns=["page_number", "text_chunk"])
    chunks_df["token_count"] = chunks_df["text_chunk"].apply(lambda x: len(x.split()) / 4)
    chunks_df = chunks_df[chunks_df["token_count"] > 30]
    
    # Generate embeddings
    text_chunks = chunks_df["text_chunk"].tolist()
    chunk_embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_numpy=True)
    chunks_df["embedding"] = list(chunk_embeddings)
    
    # Save processed data
    save_path = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
    chunks_df.to_csv(save_path, index=False)

    return jsonify({"message": "File uploaded and processed successfully!", "file_name": file.filename})

def load_text_chunks_and_embeddings():
    """Loads the stored text chunks and embeddings."""
    csv_path = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ").astype(np.float32))
    embeddings = torch.tensor(np.vstack(df["embedding"].tolist()), dtype=torch.float32, device=device)
    
    return df.to_dict(orient="records"), embeddings

@app.route("/query", methods=["POST"])
def query():
    """Handles user queries and retrieves relevant text chunks."""
    data = request.get_json()
    query_text = data.get("query")

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    pages_and_chunks, embeddings = load_text_chunks_and_embeddings()
    if pages_and_chunks is None or embeddings is None:
        return jsonify({"error": "No processed data found. Upload a PDF first!"}), 400

    with torch.no_grad():
        query_embedding = torch.tensor(
            embedding_model.encode(query_text, convert_to_tensor=False, normalize_embeddings=True),
            dtype=torch.float32, device=device
        )
        scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_scores, indices = torch.topk(scores, k=5)

    results = []
    for score, idx in zip(top_scores.tolist(), indices.tolist()):
        results.append({
            "score": round(score, 4),
            "text_chunk": pages_and_chunks[idx]["text_chunk"],
            "page_number": pages_and_chunks[idx]["page_number"]
        })

    return jsonify({"query": query_text, "results": results})

@app.route("/generate", methods=["POST"])
def generate_response():
    """Generates an AI response based on the query using the LLM."""
    data = request.get_json()
    query_text = data.get("query")

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    inputs = tokenizer(query_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=100)

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"query": query_text, "response": response_text})

if __name__ == "__main__":
    app.run(debug=True)