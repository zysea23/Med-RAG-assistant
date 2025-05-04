"""
Text embedding functionality using sentence-transformers.
"""

import logging
import torch
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Wrapper for sentence-transformers models that generate text embeddings.
    
    This class handles loading and using models for text vectorization,
    which is essential for semantic search in RAG systems.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Hugging Face model ID or path for the embedding model
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                logger.info("Moving embedding model to GPU")
                self.model.to('cuda')
                self.device = 'cuda'
            else:
                logger.info("GPU not available, using CPU")
                self.device = 'cpu'
                
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 8, 
               show_progress: bool = True,
               normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Number of texts to process at once
            show_progress: Whether to show a progress bar
            normalize: Whether to normalize the embeddings
            
        Returns:
            NumPy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings