"""
Module for searching images using text descriptions
"""

import os
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
from typing import List, Tuple


class ImageDatabase:
    def __init__(self):
        # Try to initialize CLIP model
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            print(f"Could not load CLIP model: {e}")
            print("Image search functionality will be disabled")
            self.model = None
            self.preprocess = None
    
    def encode_images(self, image_dir):
        """Encode all images in a directory"""
        encoded_images = {}
        
        if not os.path.exists(image_dir):
            print(f"Image directory {image_dir} does not exist")
            return encoded_images
            
        if self.model is None:
            print("CLIP model not available, skipping image encoding")
            return encoded_images
            
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                try:
                    image = Image.open(image_path)
                    # Skip RGBA images as CLIP expects RGB
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    
                    processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        image_features = self.model.encode_image(processed_image)
                        # Normalize once on torch to keep numeric stability
                        image_features = F.normalize(image_features, dim=-1)
                        encoded_images[image_path] = image_features.cpu().numpy()[0].astype(np.float32)
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
        
        return encoded_images
    
    def search(self, query, image_dir="data/images", top_k=1):
        """Search for images matching a text description"""
        # Encode all images
        encoded_images = self.encode_images(image_dir)
        
        if not encoded_images:
            print("No images found to search or CLIP model not available")
            return []
        
        # Encode text query
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)  # Normalize once
        
        # Calculate similarities
        similarities = {}
        text_features_np = text_features.cpu().numpy()[0]
        
        for image_path, image_features in encoded_images.items():
            # image_features already normalized
            similarity = float(np.dot(image_features, text_features_np))
            similarities[image_path] = similarity
        
        # Sort by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return sorted_similarities[:top_k]


def process(query):
    """Process an image search query"""
    # Initialize database
    db = ImageDatabase()
    
    # Perform search
    results = db.search(query)
    
    # Display results
    if results:
        print(f"Found {len(results)} images:")
        for i, (image_path, similarity) in enumerate(results):
            print(f"{i+1}. {os.path.basename(image_path)} (Similarity: {1-similarity:.4f})")
            print(f"   Path: {image_path}")
    else:
        print("No images found matching your query or CLIP model not available.")