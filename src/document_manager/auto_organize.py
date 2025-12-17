"""
Module for auto-organizing papers with automatic topic discovery
Uses clustering to automatically identify topics from paper content
"""

import os
import shutil
import pdfplumber
import numpy as np
from typing import List, Tuple
from collections import Counter
import re


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


def extract_keywords(text, top_n=3):
    """Extract top keywords from text using simple frequency analysis"""
    # Remove common words and clean text
    text = text.lower()
    words = re.findall(r'\b[a-z]{4,}\b', text)
    
    # Common stop words to filter out
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'been', 'were', 'which',
        'their', 'would', 'there', 'could', 'about', 'also', 'when', 'where',
        'what', 'such', 'these', 'those', 'than', 'then', 'them', 'into',
        'more', 'some', 'other', 'only', 'time', 'very', 'just', 'make',
        'each', 'like', 'most', 'well', 'work', 'first', 'after', 'good',
        'through', 'using', 'used', 'paper', 'figure', 'table', 'show',
        'method', 'result', 'approach', 'based', 'propose', 'present'
    }
    
    # Filter and count
    filtered_words = [w for w in words if w not in stop_words and len(w) > 4]
    word_counts = Counter(filtered_words)
    
    return [word for word, _ in word_counts.most_common(top_n)]


def generate_topic_name(texts, indices):
    """Generate a topic name from a cluster of texts"""
    # Combine texts from cluster
    combined_text = " ".join([texts[i] for i in indices])
    
    # Extract keywords
    keywords = extract_keywords(combined_text, top_n=5)
    
    if not keywords:
        return "未分类"
    
    # Use top 2-3 keywords as topic name
    return "_".join(keywords[:2])


def auto_organize_papers(source_dir, target_dir, n_clusters=None):
    """
    Auto-organize papers with automatic topic discovery
    
    Args:
        source_dir: Directory containing unorganized PDF files
        target_dir: Base directory where organized papers will be stored
        n_clusters: Number of clusters (if None, auto-detect)
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Load model
    print("Loading embedding model...")
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
            model = SentenceTransformer(local_model_path)
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files")
    print("Extracting text and generating embeddings...")
    
    # Extract texts and generate embeddings
    texts = []
    valid_files = []
    embeddings_list = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")
        
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"    Skipped (no text)")
            continue
        
        try:
            embedding = model.encode(text)
            embeddings_list.append(embedding)
            texts.append(text)
            valid_files.append(pdf_path)
        except Exception as e:
            print(f"    Error: {e}")
    
    if len(valid_files) < 2:
        print("Not enough valid files for clustering")
        return
    
    embeddings = np.array(embeddings_list)
    print(f"\nSuccessfully processed {len(valid_files)} files")
    
    # Auto-detect number of clusters if not specified
    if n_clusters is None:
        # Use elbow method heuristic: sqrt(n/2)
        n_clusters = max(2, min(5, int(np.sqrt(len(valid_files) / 2))))
        print(f"Auto-detected {n_clusters} clusters")
    else:
        n_clusters = min(n_clusters, len(valid_files))
        print(f"Using {n_clusters} clusters")
    
    # Perform clustering
    print("\nPerforming clustering...")
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
    except ImportError:
        print("scikit-learn not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-learn'])
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
    
    # Generate topic names for each cluster
    print("\nGenerating topic names...")
    cluster_topics = {}
    for cluster_id in range(n_clusters):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        topic_name = generate_topic_name(texts, indices)
        cluster_topics[cluster_id] = topic_name
        print(f"  Cluster {cluster_id}: {topic_name} ({len(indices)} papers)")
    
    # Organize files
    print("\nOrganizing files...")
    organized_count = 0
    
    for i, (pdf_path, cluster_id) in enumerate(zip(valid_files, cluster_labels)):
        filename = os.path.basename(pdf_path)
        topic_name = cluster_topics[cluster_id]
        
        # Create topic directory
        topic_dir = os.path.join(target_dir, topic_name)
        os.makedirs(topic_dir, exist_ok=True)
        
        # Move file
        destination = os.path.join(topic_dir, filename)
        
        # Handle duplicates
        if os.path.exists(destination):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(destination):
                new_filename = f"{base}_{counter}{ext}"
                destination = os.path.join(topic_dir, new_filename)
                counter += 1
        
        try:
            shutil.move(pdf_path, destination)
            print(f"  {filename} → {topic_name}/")
            organized_count += 1
        except Exception as e:
            print(f"  Error moving {filename}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Auto-organization complete!")
    print(f"  Organized {organized_count} files into {n_clusters} topics:")
    for cluster_id, topic_name in cluster_topics.items():
        count = sum(1 for label in cluster_labels if label == cluster_id)
        print(f"    - {topic_name}: {count} files")
    print("=" * 60)


def process(source_dir, n_clusters=None):
    """Process auto-organize command"""
    target_dir = "data/papers"
    auto_organize_papers(source_dir, target_dir, n_clusters)
