import os
import pymongo
from typing import List

# PDF extraction libraries
from pypdf import PdfReader

# Text splitting and embedding libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from pymongo import MongoClient


class PDFVectorizer:
    def __init__(self, 
                 pdf_path: str,
                 mongo_uri: str = "mongodb+srv://pranay:sih2024@cluster0.kx5kz.mongodb.net/",
                 db_name: str = "SIH",
                 collection_name: str = "triaL"):
        self.pdf_path = pdf_path
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def process_pdf(self):
        text = self.extract_text()
        chunks = self.split_text(text)
        embeddings = self.embed_chunks(chunks)
        self.store_embeddings(chunks, embeddings)

    def extract_text(self) -> str:
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def split_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(text)

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        return self.model.encode(chunks).tolist()

    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]]):
        documents = [
            {
                "text": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.collection.insert_many(documents)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def cosine_filename(a: List[float], b: List[float]) -> str:
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
def search_pdf_chunks(query: str, top_k: int = 5) -> List[dict]:
    client = MongoClient("mongodb+srv://pranay:sih2024@cluster0.kx5kz.mongodb.net/")
    db = client["SIH"]
    collection = db["common"]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()

    # Retrieve all documents with embeddings
    documents = collection.find({"embedding": {"$exists": True}}, {"text": 1, "embedding": 1,'docName':1,'type':1})
    # print(f"\n\n\n\n\n{documents}")
    # Compute similarity
    results = []
    for doc in documents:
        # print(f"n\n\n\n\n\{doc}")
        similarity = cosine_similarity(query_embedding, doc['embedding'])
        results.append({
            "text": doc['text'],
            "similarity": similarity,
            "source" : doc['docName']+'.'+doc['type']
        })

    # Sort by similarity and get top_k results
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


# Example usage
if __name__ == "__main__":
    # Process a PDF
    pdf_path = '/workspaces/SnackOverflow-SIH/backend/General College Information (1).pdf'  # Update the path if necessary
    pdf_vectorizer = PDFVectorizer(pdf_path=pdf_path)
    pdf_vectorizer.process_pdf()
    
    print("PDF processing completed and embeddings stored.\n")
    
    # Search PDF chunks
    query = "What is the main topic of this document?"
    results = search_pdf_chunks(query)
    
    if results:
        for idx, result in enumerate(results, 1):
            print(f"Result {idx}:")
            print(f"Chunk: {result.get('text', 'N/A')}")
            print(f"Similarity Score: {result.get('similarity', 'N/A'):.4f}\n")
    else:
        print("No results found.")