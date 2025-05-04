"""
Text processing utilities for chunking and preparing text for RAG systems.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 100,
    min_chunk_length: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in words
        chunk_overlap: Number of words to overlap between chunks
        min_chunk_length: Minimum length of a chunk in words to be included
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split text into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Prepare to process paragraphs into chunks
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        # Split paragraph into words and get word count
        words = para.split()
        para_size = len(words)
        
        # If a single paragraph exceeds chunk size, split it further
        if para_size > chunk_size * 1.5:
            # Process large paragraph separately
            for i in range(0, para_size, chunk_size - chunk_overlap):
                para_chunk = ' '.join(words[i:i + chunk_size])
                if len(para_chunk.split()) >= min_chunk_length:
                    chunks.append(para_chunk)
        else:
            # For normal paragraphs, try to combine them up to chunk_size
            if current_size + para_size <= chunk_size:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size
            else:
                # Current chunk is full, finalize it if it's long enough
                if current_size >= min_chunk_length:
                    chunks.append(' '.join(current_chunk))
                
                # Start a new chunk with this paragraph
                current_chunk = [para]
                current_size = para_size
    
    # Don't forget the last chunk
    if current_chunk and current_size >= min_chunk_length:
        chunks.append(' '.join(current_chunk))
    
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, normalizing quotes, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Remove page numbers and headers/footers (common in PDFs)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    return text