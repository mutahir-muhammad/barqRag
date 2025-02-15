# 📄 Flask-Based Retrieval-Augmented Generation (RAG) API

This is a **Flask-based Python API** that enables users to:
- Upload a **PDF**
- Process its content into **chunks**
- Generate **embeddings**
- Use a **Retrieval-Augmented Generation (RAG) pipeline** to answer queries
- Integrate a **language model (LLM)** for AI-generated responses

---

## **1️⃣ Importing Required Libraries**

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF (for PDF processing)
import torch  # PyTorch for embeddings and LLM computations
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util  # For embedding generation and similarity search
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face transformers for LLM
from tqdm.auto import tqdm  # Progress bar utility
from spacy.lang.en import English  # NLP for sentence tokenization
import logging  # Logging for debugging
```

### **Purpose of these libraries:**
- **Flask & Flask-CORS**: Creates a web API that handles HTTP requests.
- **fitz (PyMuPDF)**: Extracts text from PDF files.
- **torch (PyTorch)**: Handles deep learning computations.
- **numpy & pandas**: Data manipulation.
- **sentence-transformers**: Converts text into embeddings.
- **transformers (Hugging Face)**: Loads and runs a pre-trained Large Language Model (LLM).
- **spaCy**: Splits text into meaningful sentences.
- **logging**: Outputs debug messages.

---

## **2️⃣ Flask App Initialization**

```python
app = Flask(__name__)
CORS(app)
```
- Initializes the Flask app.
- Enables **CORS** for frontend-backend communication.

---

## **3️⃣ Logging Setup**

```python
logging.basicConfig(level=logging.DEBUG)
```
- Enables **DEBUG level** logging for detailed output.

---

## **4️⃣ Model Initialization**

```python
device = "cpu"
```
- Runs models **on CPU only** (per project constraint).

### **Loading Sentence Transformer Model**

```python
logging.info("Loading sentence transformer model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
```
- Loads a **pre-trained Sentence Transformer** model for **embedding generation**.

### **Loading Tokenizer & LLM**

```python
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
logging.info("Loading LLM model...")
llm_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={"": "cpu"}
)
```
- Loads **Google's Gemma-2B-IT model** for generating responses.
- Uses `low_cpu_mem_usage=True` to optimize memory usage.

---

## **5️⃣ Setting Up File Upload Directory**

```python
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
```
- Creates an **"uploads"** directory if it doesn’t exist.

---

## **6️⃣ PDF Processing Functions**

### **Extract Text from PDF**

```python
def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text").replace("\n", " ").strip() for page in tqdm(doc, desc="Extracting PDF text")]
    if not any(pages):
        raise ValueError("No text extracted from PDF.")
    return pages
```
- Extracts text from each page of a **PDF file**.

### **Sentence Splitting Using spaCy**

```python
nlp = English()
nlp.add_pipe("sentencizer")

def split_sentences(text):
    return [str(sent).strip() for sent in nlp(text).sents if str(sent).strip()]
```
- Splits text into **meaningful sentences**.

### **Chunking Sentences**

```python
def chunk_sentences(sentences, chunk_size=10):
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
```
- Groups **10 sentences together** to form a chunk.

---

## **7️⃣ PDF Upload API**

```python
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)
```
- Accepts **PDF file uploads** from frontend.
- Saves them in the `uploads/` folder.

### **Processing the Uploaded PDF**

```python
pages_text = read_pdf(pdf_path)
pages_sentences = [split_sentences(page) for page in pages_text]
pages_chunks = [chunk_sentences(sentences) for sentences in pages_sentences]
flat_chunks = [(page_no + 1, chunk) for page_no, chunks in enumerate(pages_chunks) for chunk in chunks]
chunks_df = pd.DataFrame(flat_chunks, columns=["page_number", "text_chunk"])
```
- **Reads** the PDF and **splits** it into chunks.

### **Embedding Generation**

```python
text_chunks = chunks_df["text_chunk"].tolist()
chunk_embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_numpy=True)
chunks_df["embedding"] = list(chunk_embeddings)
```
- Converts each **text chunk into embeddings** for fast retrieval.

### **Saving Processed Data**

```python
save_path = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
chunks_df.to_csv(save_path, index=False)
```
- Saves **chunked text + embeddings** into a CSV file.

---

## **8️⃣ Query API**

```python
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=False, normalize_embeddings=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_scores, indices = torch.topk(scores, k=5)
```
- **Finds the most relevant** text chunks for a user query.

---

## **9️⃣ Generative AI Response API**

```python
@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.get_json()
    query_text = data.get("query")
    
    inputs = tokenizer(query_text, return_tensors="pt").to(device)
    output = llm_model.generate(**inputs, max_new_tokens=100)
    
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"query": query_text, "response": response_text})
```
- Uses the **LLM (Gemma-2B-IT)** to **generate a response**.

---

## **🔚 Final Thoughts**
This API:
✅ Uploads and processes PDFs
✅ Creates embeddings for **fast retrieval**
✅ Supports **AI-powered chat**

Would you like any modifications or optimizations? 🚀

