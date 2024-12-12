import os
import ssl
import asyncio
import logging
import traceback
from flask import Flask, request, jsonify, send_file
from flask_pymongo import PyMongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from playwright.async_api import async_playwright
from fpdf import FPDF
from dotenv import load_dotenv
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import pandas as pd
from sidebar import sidebar_bp, init_mongo
from summary import summary_bp
from admin_page import admin_page
from chat_model import *

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
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = "your_unique_secret_key"

# MongoDB configuration
try:
    client = MongoClient(
        MONGODB_URI,
        tls=True,
        tlsAllowInvalidCertificates=True
    )
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    init_mongo(app)
    print("Connected to MongoDB")
except ConnectionFailure:
    print("Failed to connect to MongoDB")

app.register_blueprint(sidebar_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(admin_page)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from a PDF file
def extract_text_from_pdf(file_storage) -> str:
    try:
        file_bytes = BytesIO(file_storage.read())
        reader = PdfReader(file_bytes)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return ""

# Extract text from a CSV file
def extract_text_from_csv(file_storage) -> str:
    try:
        file_stream = StringIO(file_storage.stream.read().decode("utf-8"))
        df = pd.read_csv(file_stream)
        return df.to_string(index=False)
    except Exception as e:
        return ""

# Cosine similarity calculation
def cosine_similarity(a: list, b: list) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# Save the content to a PDF
def save_to_pdf(content, filename="output.pdf"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.add_font('ArialUnicode', '', 'C:/Windows/Fonts/arialuni.ttf', uni=True)
        pdf.set_font('ArialUnicode', '', 12)

        lines = content.split("\n")
        for line in lines:
            pdf.cell(200, 10, txt=line, ln=True)

        os.makedirs('outputs', exist_ok=True)
        full_path = os.path.join('outputs', filename)
        pdf.output(full_path)
        return full_path
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return ""

# Scrape and transform content using Playwright
async def scrape_and_transform(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        print(f"Error during scraping: {e}")
        return ""

@app.route('/scrape_and_generate_pdf', methods=['POST'])
async def scrape_and_generate_pdf():
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        content = await scrape_and_transform(url)
        if not content.strip():
            return jsonify({"error": "No content found or an error occurred during scraping."}), 400

        pdf_filename = f"scraped_content_{hash(url)}.pdf"
        pdf_path = save_to_pdf(content, pdf_filename)

        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_type = filename.rsplit('.', 1)[1].lower()
        doc_name = filename.rsplit('.', 1)[0]

        if file_type == 'pdf':
            text = extract_text_from_pdf(file)
        elif file_type == 'csv':
            text = extract_text_from_csv(file)
        else:
            return jsonify({"error": "Unsupported file type."}), 400

        if not text:
            return jsonify({"error": "Failed to extract text from the file."}), 400

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        if not chunks:
            return jsonify({"error": "Failed to create text chunks."}), 400

        embeddings = model.encode(chunks).tolist()

        documents = [
            {
                "docName": doc_name,
                "type": file_type,
                "text": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        try:
            collection.insert_many(documents)
            return jsonify({"message": f"File '{filename}' uploaded and processed successfully."}), 200
        except Exception as e:
            return jsonify({"error": "Failed to store embeddings in the database."}), 500
    else:
        return jsonify({"error": "File type not allowed. Only PDF and CSV are supported."}), 400

@app.route('/api/documents', methods=['GET'])
def get_documents():
    try:
        pipeline = [
            {"$group": {"_id": {"docName": "$docName", "type": "$type"}}},
            {"$project": {"docName": "$_id.docName", "type": "$_id.type", "_id": 0}}
        ]
        unique_documents = list(collection.aggregate(pipeline))
        return jsonify(unique_documents), 200
    except Exception as e:
        return jsonify({"error": "Failed to fetch documents."}), 500

@app.route('/api/documents/<doc_name>', methods=['DELETE'])
def delete_document(doc_name):
    try:
        result = collection.delete_many({"docName": doc_name})
        if result.deleted_count == 0:
            return jsonify({"message": f"No documents found for document '{doc_name}'."}), 404
        return jsonify({"message": f"Document '{doc_name}' and its chunks deleted successfully."}), 200
    except Exception as e:
        return jsonify({"error": "Failed to delete document."}), 500

@app.route('/api/search', methods=['POST'])
def search_chunks():
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
    except Exception as e:
        return jsonify({"error": "An error occurred during the search."}), 500

@app.route('/api/recent-questions', methods=['GET'])
def get_recent_questions():
    recent_questions = list(db.recent_questions.find().sort('timestamp', -1).limit(3))
    formatted_questions = [
        {
            'question': question['question'],
            'time': (datetime.utcnow() - question['timestamp']).total_seconds() / 3600
        } for question in recent_questions
    ]
    return jsonify(formatted_questions)

@socketio.on('send_message')
def handle_send_message(msg):
    print("Generating response for: ", msg.get("msg", ""))
    chat_id = msg.get("id", "default_id")
    user_message = msg.get("msg", "")
    message_history = msg.get("messages", [])
    db.chatbot_requests.insert_one({
        'chat_id': chat_id,
        'message': user_message,
        'timestamp': datetime.utcnow()
    })
    db.recent_questions.insert_one({
        'question': user_message,
        'timestamp': datetime.utcnow()
    })
    try:
        res = ChatModel(chat_id, user_message, message_history)
        print("sending", res)
        emit("response", res)
    except Exception as e:
        print(f"Error in ChatModel: {str(e)}")
        emit("response", {"error": "An error occurred while processing your message."})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response

if __name__ == '__main__':
    socketio.run(app, use_reloader=False, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)