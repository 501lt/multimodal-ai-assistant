"""
Module for adding and classifying papers
"""

import os
import shutil
from sentence_transformers import SentenceTransformer
import pdfplumber
from typing import List


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle None case
    return text


def classify_paper(pdf_path, topics):
    """Classify a paper based on its content and given topics"""
    # Force offline mode
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # Specify the local path explicitly
    local_model_path = os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    
    try:
        # Try to load model with explicit local path
        print(f"Attempting to load model from: {local_model_path}")
        model = SentenceTransformer(local_model_path)
    except Exception as e:
        print(f"Failed to load model from local path: {e}")
        # Fallback to named model (will use cache if available)
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e2:
            print(f"Failed to load model by name: {e2}")
            raise Exception("Could not load any SentenceTransformer model. Please ensure you have downloaded the model files to the correct location.")
    
    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)
    
    # Create embeddings for paper text and topics
    embeddings = model.encode([paper_text] + topics)
    paper_embedding = embeddings[0]
    topic_embeddings = embeddings[1:]
    
    # Calculate similarities
    import numpy as np
    similarities = np.zeros(len(topic_embeddings))
    for i, topic_emb in enumerate(topic_embeddings):
        # Cosine similarity calculation
        sim = np.dot(paper_embedding, topic_emb) / (np.linalg.norm(paper_embedding) * np.linalg.norm(topic_emb))
        similarities[i] = sim
    
    # Get the most similar topic
    best_match_idx = np.argmax(similarities)
    best_topic = topics[best_match_idx]
    
    return best_topic


def process(pdf_path, topics):
    """Process a paper: classify and move to appropriate folder"""
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    # Create papers directory if it doesn't exist
    papers_dir = "data/papers"
    os.makedirs(papers_dir, exist_ok=True)
    
    # Classify the paper
    try:
        best_topic = classify_paper(pdf_path, topics)
        print(f"Classified paper as: {best_topic}")
        
        # Create topic directory if it doesn't exist
        topic_dir = os.path.join(papers_dir, best_topic)
        os.makedirs(topic_dir, exist_ok=True)
        
        # Move the paper to the topic directory
        filename = os.path.basename(pdf_path)
        destination = os.path.join(topic_dir, filename)
        shutil.move(pdf_path, destination)
        
        print(f"Moved {filename} to {destination}")
    except Exception as e:
        print(f"Error processing paper: {str(e)}")
        print("Please ensure you have downloaded the required models to:")
        print("C:\\Users\\李腾\\.cache\\torch\\sentence_transformers\\sentence-transformers_all-MiniLM-L6-v2\\")