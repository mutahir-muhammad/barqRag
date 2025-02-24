from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pymongo import MongoClient

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# MongoDB Connection
MONGO_URI = "mongodb+srv://myAtlasDBUser:<enter your password for the cluster>@myatlasclusteredu.fjdg2os.mongodb.net/my_database?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
# The lines `db = client["my_database"]` and `collection = db["processed_data"]` are establishing a
# connection to a MongoDB database using the PyMongo library.
db = client["my_database"]
collection = db["processed_data"]

# Configure logging
logging.basicConfig(level=logging.INFO)

# Ensure CPU usage
device = "cpu"

# Load models
logging.info("Loading sentence transformer model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
logging.info("Loading tokenizer and LLM model...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
llm_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)

# The code snippet `UPLOAD_FOLDER = "uploads"` defines a variable `UPLOAD_FOLDER` with the value
# "uploads", which is the name of the folder where uploaded PDF files will be stored.
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    return [page.get_text("text").replace("\n", " ").strip() for page in doc]

def generate_embeddings(chunks):
    """Generates sentence embeddings."""
    return embedding_model.encode(chunks, batch_size=32, convert_to_numpy=True).tolist()

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid file format. Only PDFs are allowed."}), 400
    
    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)
    
    pages_text = extract_text_from_pdf(pdf_path)
    if not pages_text:
        return jsonify({"error": "No text extracted from PDF."}), 400
    
    chunks = [(i+1, text) for i, text in enumerate(pages_text) if len(text.split()) > 30]
    df = pd.DataFrame(chunks, columns=["page_number", "text_chunk"])
    df["embedding"] = generate_embeddings(df["text_chunk"].tolist())
    
    collection.insert_many(df.to_dict(orient="records"))
    return jsonify({"message": "File uploaded and processed successfully!", "file_name": file.filename})

def retrieve_relevant_chunks(query_text):
    """Finds the most relevant text chunks from MongoDB."""
    stored_data = list(collection.find({}, {"_id": 0}))
    if not stored_data:
        return [], None
    
    df = pd.DataFrame(stored_data)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
    scores = util.pytorch_cos_sim(torch.tensor(query_embedding, dtype=torch.float32), torch.tensor(df["embedding"].tolist(), dtype=torch.float32))[0]
    top_indices = scores.argsort(descending=True)[:3]
    retrieved_chunks = df.iloc[top_indices][["page_number", "text_chunk"]].to_dict(orient="records")
    return retrieved_chunks, " ".join([chunk["text_chunk"] for chunk in retrieved_chunks])

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    retrieved_chunks, context_text = retrieve_relevant_chunks(query_text)
    if not retrieved_chunks:
        return jsonify({"error": "No relevant data found."}), 400
    
    inputs = tokenizer(context_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=100)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return  response_text

if __name__ == "__main__":
    app.run(debug=True)
