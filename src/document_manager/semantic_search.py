"""
Module for fine-grained semantic search with paper chunks and page numbers.
Supports searching for specific passages and identifying their locations in papers.
"""

import os
import re
import chromadb
import pdfplumber
from typing import List, Tuple
import numpy as np

# Optional PDF rasterization for OCR
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# Try to import OCR support
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class ChunkBasedSearch:
    """Fine-grained semantic search with chunks and page information"""
    
    def __init__(self, chunk_size=800, overlap=100):
        """
        Initialize the search engine.
        
        Args:
            chunk_size: Characters per chunk
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Set environment variables for offline usage
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        self.client = chromadb.Client()
        
        # Try to load SentenceTransformer from local cache or fallback to simple encoder
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
                print(f"Loading SentenceTransformer from: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
            else:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            print("Using simple text encoder due to model loading issues")
            from .search_paper import SimpleTextEncoder
            self.model = SimpleTextEncoder()
        
        # Determine embedding dimension and use it for collection name
        try:
            probe = self.model.encode("probe")
            probe_arr = np.asarray(probe)
            if probe_arr.ndim > 1:
                dim = int(probe_arr.shape[-1])
            else:
                dim = int(probe_arr.shape[0])
        except Exception:
            dim = 256
        
        self.collection_name = f"chunks_d{dim}"
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
    
    def extract_text_with_pages(self, pdf_path) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page information.
        Falls back to OCR if native extraction is too short.
        
        Returns:
            List of (text, page_number) tuples
        """
        text_with_pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""

                    # Fallback OCR when page text is too short (likely scanned)
                    if len(text.strip()) < 50 and OCR_AVAILABLE:
                        ocr_text = ""
                        try:
                            if PDF2IMAGE_AVAILABLE:
                                images = convert_from_path(
                                    pdf_path, dpi=300, first_page=page_num, last_page=page_num
                                )
                                if images:
                                    pil_image = images[0]
                                    try:
                                        pil_image = pil_image.convert('L')
                                    except Exception:
                                        pass
                                    ocr_text = pytesseract.image_to_string(pil_image, lang='eng+chi_sim')
                            else:
                                # Best-effort fallback using pdfplumber rendering
                                img = page.to_image(resolution=300).original
                                try:
                                    img = img.convert('L')
                                except Exception:
                                    pass
                                ocr_text = pytesseract.image_to_string(img, lang='eng+chi_sim')
                        except Exception as ocr_e:
                            print(f"  OCR failed for page {page_num}: {ocr_e}")

                        if len((ocr_text or '').strip()) > len(text.strip()):
                            text = ocr_text

                    # Basic cleanup
                    if text:
                        # merge hyphenated line breaks and normalize whitespace
                        text = re.sub(r"-\n", "", text)
                        text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        text_with_pages.append((text, page_num))
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text_with_pages
    
    def create_chunks(self, text: str, page_num: int) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks with page number.
        
        Returns:
            List of (chunk_text, page_number) tuples
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:  # Only keep non-empty chunks
                chunks.append((chunk, page_num))
            start += self.chunk_size - self.overlap
        return chunks
    
    def index_paper(self, file_path: str):
        """Index all chunks from a paper"""
        print(f"Indexing paper: {os.path.basename(file_path)}")
        
        # Remove previous chunks for this file to avoid duplicates
        try:
            self.collection.delete(where={"file_path": file_path})
        except Exception:
            pass

        text_with_pages = self.extract_text_with_pages(file_path)
        if not text_with_pages:
            print(f"  No text extracted from {file_path}")
            return
        
        filename = os.path.basename(file_path)
        chunk_id_counter = 0
        
        for text, page_num in text_with_pages:
            chunks = self.create_chunks(text, page_num)
            for chunk_text, page in chunks:
                try:
                    # Generate embedding
                    raw_embedding = self.model.encode(chunk_text)
                    embedding = self._to_vector(raw_embedding)
                    
                    if np.linalg.norm(embedding) == 0:
                        continue
                    
                    # Create unique ID for this chunk
                    chunk_id = f"{filename}_chunk_{chunk_id_counter}"
                    chunk_id_counter += 1
                    
                    # Store chunk with metadata
                    self.collection.add(
                        documents=[chunk_text],
                        embeddings=[embedding],
                        ids=[chunk_id],
                        metadatas=[{
                            "file_path": file_path,
                            "page_num": page,
                            "filename": filename
                        }]
                    )
                except Exception as e:
                    print(f"  Error processing chunk from {filename} page {page}: {e}")
        
        print(f"  Indexed {chunk_id_counter} chunks from {filename}")
    
    def search(self, query: str, n_results: int = 1) -> List[Tuple[str, str, int, float]]:
        """
        Search for query across all indexed chunks.
        
        Returns:
            List of (filename, chunk_text, page_num, similarity) tuples
        """
        try:
            # Generate embedding for query
            raw_query_embedding = self.model.encode(query)
            query_embedding = self._to_vector(raw_query_embedding)
            
            if np.linalg.norm(query_embedding) == 0:
                print("Query has no content; cannot compute embedding.")
                return []
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results: convert distance to similarity
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for doc, distance, metadata in zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                ):
                    similarity = float(distance)
                    filename = metadata.get('filename', 'Unknown')
                    page_num = metadata.get('page_num', 0)
                    formatted_results.append((filename, doc, page_num, similarity))
            
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def index_all_papers(self, papers_dir: str = "data/papers"):
        """Index all PDF files in a directory"""
        if not os.path.exists(papers_dir):
            print(f"Papers directory {papers_dir} not found")
            return
        
        pdf_count = 0
        for root, dirs, files in os.walk(papers_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    try:
                        self.index_paper(file_path)
                        pdf_count += 1
                    except Exception as e:
                        print(f"Error indexing {file_path}: {e}")
        
        print(f"\nCompleted indexing {pdf_count} papers.")


def process(query: str):
    """Process a semantic search query with chunks and page numbers"""
    print(f"Starting fine-grained semantic search for: '{query}'")
    print("-" * 60)
    
    # Initialize search engine
    search_engine = ChunkBasedSearch()
    
    # Index all papers
    search_engine.index_all_papers()
    
    # Perform search (get top 5 results so we can pick a substantive passage)
    results = search_engine.search(query, n_results=5)
    
    # Helpers to decide whether to synthesize an answer
    def looks_like_url(text: str) -> bool:
        if not text:
            return True
        url_pattern = re.compile(r"https?://|[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        return bool(url_pattern.search(text))
    
    # Display result (pick first substantive passage among top results)
    if results:
        chosen = None
        for rec in results:
            fname, ch, pnum, sim = rec
            if ch and (len(ch.strip()) >= 120) and not looks_like_url(ch):
                chosen = rec
                break
        if not chosen:
            chosen = results[0]

        filename, chunk, page_num, similarity = chosen
        print(f"\nMost relevant passage:\n")
        print(f"File: {filename}")
        print(f"Page: {page_num}")
        print(f"Similarity: {similarity:.4f}")
        print(f"\nContent (passage):\n{chunk}")
    else:
        print("No relevant passages found.")
