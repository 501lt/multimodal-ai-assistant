"""
Module for searching papers using semantic search
"""

import os
import chromadb
import pdfplumber
from typing import List, Tuple
import numpy as np


class SimpleTextEncoder:
    """A simple text encoder for offline usage"""
    def __init__(self):
        pass
    
    def encode(self, texts):
        """Simple encoding method using character n-grams"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Hash characters (including中文) into 256 buckets
            vector = np.zeros(256, dtype=float)
            for char in text.lower():
                if char.isspace():
                    continue
                bucket = ord(char) % 256
                vector[bucket] += 1.0
            
            # Normalize
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            embeddings.append(vector)
        
        return np.array(embeddings)
    
    def similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        if len(vec2.shape) == 1:
            vec2 = vec2.reshape(1, -1)
        
        similarities = np.dot(vec1, vec2.T)
        return similarities


class PaperDatabase:
    def __init__(self):
        # Set environment variables for offline usage
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        self.client = chromadb.Client()

        # Try to initialize SentenceTransformer from local cache first, fallback to name
        try:
            from sentence_transformers import SentenceTransformer
            local_model_path = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "torch",
                "sentence_transformers",
                "sentence-transformers_all-MiniLM-L6-v2",
            )
            if os.path.isdir(local_model_path):
                print(f"Attempting to load SentenceTransformer from local path: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
            else:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            print("Using simple text encoder due to model loading issues")
            self.model = SimpleTextEncoder()

        # Determine embedding dimension and use dimension-based collection name to avoid mixing
        try:
            probe = self.model.encode("probe")
            probe_arr = np.asarray(probe)
            if probe_arr.ndim > 1:
                dim = int(probe_arr.shape[-1])
            else:
                dim = int(probe_arr.shape[0])
        except Exception:
            dim = 256  # fallback for SimpleTextEncoder

        self.collection_name = f"papers_d{dim}"
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError:
            self.collection = self.client.create_collection(self.collection_name)

    @staticmethod
    def _to_vector(embedding) -> list:
        """Normalize embedding output into a flat list of floats"""
        arr = np.asarray(embedding)
        if arr.ndim > 1:
            arr = arr[0]
        return arr.tolist()
    
    def index_paper(self, file_path):
        """Index a paper in the database"""
        # Extract text from PDF
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Generate embedding
            raw_embedding = self.model.encode(text)
            embedding = self._to_vector(raw_embedding)
            if np.linalg.norm(embedding) == 0:
                print(f"Warning: No alphabetic content in {file_path}, skipping indexing")
                return
            
            # Add to collection
            filename = os.path.basename(file_path)
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[filename],
                metadatas=[{"file_path": file_path}]
            )
            print(f"Indexed paper: {filename}")
        except Exception as e:
            print(f"Error indexing paper {file_path}: {str(e)}")
    
    def search(self, query, n_results=3):
        """Search for papers based on a query"""
        try:
            # Generate embedding for query
            raw_query_embedding = self.model.encode(query)
            query_embedding = self._to_vector(raw_query_embedding)
            if np.linalg.norm(query_embedding) == 0:
                print("Query has no alphabetic content; cannot compute embedding.")
                return []
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Return results
            return list(zip(results['ids'][0], results['distances'][0], results['metadatas'][0]))
        except Exception as e:
            print(f"Error searching papers: {str(e)}")
            return []


def process(query):
    """Process a paper search query"""
    # Initialize database
    db = PaperDatabase()
    
    # Scan data/papers directory for PDF files to index
    papers_dir = "data/papers"
    if os.path.exists(papers_dir):
        for root, dirs, files in os.walk(papers_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    # Check if already indexed
                    try:
                        existing = db.collection.get(ids=[file])
                        if len(existing['ids']) == 0:  # Not indexed yet
                            db.index_paper(file_path)
                    except:
                        db.index_paper(file_path)
    
    # Perform search
    results = db.search(query)
    
    # Display results
    if results:
        print(f"Found {len(results)} papers:")
        for i, (id, distance, metadata) in enumerate(results):
            print(f"{i+1}. {id} (Similarity: {distance:.4f})")
            print(f"   Path: {metadata['file_path']}")
    else:
        print("No papers found matching your query.")