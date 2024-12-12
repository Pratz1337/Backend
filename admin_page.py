import os
import ssl
import certifi
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from dotenv import load_dotenv
from flask_cors import CORS

# PDF and embedding libraries
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from werkzeug.utils import secure_filename
from flask import Blueprint

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = ["MONGODB_URI", "DATABASE_NAME", "COLLECTION_NAME", "INDEX_NAME"]
missing_vars = [var for var in required_vars if var not in os.environ]

if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Get environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize Flask app
admin_page = Blueprint('admin_page', __name__)
CORS(admin_page)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize MongoDB client with updated TLS parameters
client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF and CSV Upload Route
@admin_page.route('/api/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload and process PDF or CSV files."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_type = filename.rsplit('.', 1)[1].lower()
        doc_name = filename.rsplit('.', 1)[0]

        # Process the file based on its type
        if file_type == 'pdf':
            text = extract_text_from_pdf(file)
        elif file_type == 'csv':
            text = extract_text_from_csv(file)
        else:
            return jsonify({"error": "Unsupported file type."}), 400

        if not text:
            return jsonify({"error": "Failed to extract text from the file."}), 400

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        if not chunks:
            return jsonify({"error": "Failed to create text chunks."}), 400

        # Generate embeddings
        embeddings = model.encode(chunks).tolist()

        # Prepare documents with metadata
        documents = [
            {
                "docName": doc_name,
                "type": file_type,
                "text": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Insert documents into MongoDB
        try:
            collection.insert_many(documents)
            return jsonify({"message": f"File '{filename}' uploaded and processed successfully."}), 200
        except Exception:
            return jsonify({"error": "Failed to store embeddings in the database."}), 500
    else:
        return jsonify({"error": "File type not allowed. Only PDF and CSV are supported."}), 400

def extract_text_from_pdf(file_storage) -> str:
    """Extract text from a PDF file uploaded as a FileStorage object."""
    try:
        file_bytes = BytesIO(file_storage.read())
        reader = PdfReader(file_bytes)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        return ""

def extract_text_from_csv(file_storage) -> str:
    """Convert CSV data to a string representation."""
    try:
        file_stream = StringIO(file_storage.stream.read().decode("utf-8"))
        df = pd.read_csv(file_stream)
        return df.to_string(index=False)
    except Exception:
        return ""

def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

@admin_page.route('/api/search', methods=['POST'])
def search_chunks():
    """Endpoint to search for relevant chunks based on a query."""
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required."}), 400

    query_embedding = model.encode(query).tolist()

    try:
        documents = collection.find({"embedding": {"$exists": True}}, {"docName": 1, "type": 1, "text": 1, "embedding": 1})

        results = []
        for doc in documents:
            similarity = cosine_similarity(query_embedding, doc['embedding'])
            results.append({
                "docName": doc.get("docName", "N/A"),
                "type": doc.get("type", "N/A"),
                "text": doc.get("text", "N/A"),
                "similarity": similarity
            })

        if not results:
            return jsonify({"message": "No documents found."}), 200

        top_k = 5
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]

        return jsonify(results), 200
    except Exception:
        return jsonify({"error": "An error occurred during the search."}), 500
