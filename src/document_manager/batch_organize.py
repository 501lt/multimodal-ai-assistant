"""
Module for batch organizing papers
Scan a directory and automatically classify and organize all PDF files
"""

import os
import shutil
from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
from typing import List


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"  Error extracting text: {e}")
    return text


def classify_paper(pdf_path, topics, model):
    """Classify a paper based on its content and given topics"""
    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)
    
    if not paper_text.strip():
        print(f"  Warning: No text extracted, skipping classification")
        return None
    
    # Create embeddings for paper text and topics
    try:
        embeddings = model.encode([paper_text] + topics)
        paper_embedding = embeddings[0]
        topic_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = np.zeros(len(topic_embeddings))
        for i, topic_emb in enumerate(topic_embeddings):
            # Cosine similarity calculation
            sim = np.dot(paper_embedding, topic_emb) / (
                np.linalg.norm(paper_embedding) * np.linalg.norm(topic_emb)
            )
            similarities[i] = sim
        
        # Get the most similar topic
        best_match_idx = np.argmax(similarities)
        best_topic = topics[best_match_idx]
        confidence = similarities[best_match_idx]
        
        return best_topic, confidence
    except Exception as e:
        print(f"  Error during classification: {e}")
        return None


def organize_papers(source_dir, target_dir, topics):
    """
    Batch organize papers from source directory to target directory by topics
    
    Args:
        source_dir: Directory containing unorganized PDF files
        target_dir: Base directory where organized papers will be stored
        topics: List of topic names for classification
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Load SentenceTransformer model
    print("Loading classification model...")
    try:
        local_model_path = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "torch",
            "sentence_transformers",
            "sentence-transformers_all-MiniLM-L6-v2",
        )
        if os.path.isdir(local_model_path):
            print(f"Loading from: {local_model_path}")
            model = SentenceTransformer(local_model_path)
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find all PDF files in source directory
    pdf_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files to organize")
    print(f"Topics: {', '.join(topics)}")
    print("-" * 60)
    
    # Process each PDF
    organized_count = 0
    skipped_count = 0
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing: {filename}")
        
        # Classify the paper
        result = classify_paper(pdf_path, topics, model)
        
        if result is None:
            print(f"  Skipped (unable to classify)")
            skipped_count += 1
            continue
        
        best_topic, confidence = result
        print(f"  Classified as: {best_topic} (confidence: {confidence:.4f})")
        
        # Create topic directory
        topic_dir = os.path.join(target_dir, best_topic)
        os.makedirs(topic_dir, exist_ok=True)
        
        # Move the paper to the topic directory
        destination = os.path.join(topic_dir, filename)
        
        # Handle duplicate filenames
        if os.path.exists(destination):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(destination):
                new_filename = f"{base}_{counter}{ext}"
                destination = os.path.join(topic_dir, new_filename)
                counter += 1
        
        try:
            shutil.move(pdf_path, destination)
            print(f"  Moved to: {destination}")
            organized_count += 1
        except Exception as e:
            print(f"  Error moving file: {e}")
            skipped_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Batch organization complete!")
    print(f"  Successfully organized: {organized_count} files")
    print(f"  Skipped: {skipped_count} files")
    print("=" * 60)


def process(source_dir, topics):
    """Process batch organization command"""
    target_dir = "data/papers"
    organize_papers(source_dir, target_dir, topics)
