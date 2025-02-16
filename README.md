# chat-pdf-api

> A Flask-based Python API for PDF processing and retrieval-augmented generation (RAG)

This is a **Flask-based Python API** that enables users to:

- Upload a **PDF**
- Process its content into **chunks**
- Generate **embeddings**
- Use a **Retrieval-Augmented Generation (RAG) pipeline** to answer queries
- Integrate a **language model (LLM)** for AI-generated responses

## Setup

Clone the repository:

```bash
git clone git@github.com:kamranahmedse/barqRag.git
pip install -r requirements.txt
python app.py
```

## Usage

API provides the following endpoints:

- `/upload` - Upload a PDF file
- `/process` - Process the uploaded PDF
- `/answer` - Answer a question using the RAG pipeline

## License

MIT (c) Muhammad Mutahir


