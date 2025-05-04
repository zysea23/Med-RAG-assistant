"""
FAISS-based vector indexing for similarity search.
"""

import logging
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorIndex:
    """
    Vector index for efficient similarity search using FAISS.
    
    This class handles storing and searching vector embeddings,
    enabling fast semantic retrieval for RAG systems.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the vector index.
        
        Args:
            dimension: Dimensionality of the embedding vectors
        """
        logger.info(f"Initializing vector index with dimension {dimension}")
        
        # Create a flat L2 index (exact search)
        try:
            if faiss.get_num_gpus() > 0:
                # Use GPU if available
                logger.info("Using GPU for vector indexing")
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatL2(res, dimension)
            else:
                # Fall back to CPU
                logger.info("Using CPU for vector indexing")
                self.index = faiss.IndexFlatL2(dimension)
                
            self.dimension = dimension
            self.vectors = []  # Store vectors for potential reuse
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: NumPy array of vectors to add (shape: [n, dimension])
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")
        
        logger.info(f"Adding {vectors.shape[0]} vectors to index")
        
        try:
            self.index.add(vectors)
            self.vectors.append(vectors)  # Store for potential reuse
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_vector: Query vector (shape: [1, dimension])
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dimension}, got {query_vector.shape[1]}")
        
        logger.debug(f"Searching for {k} neighbors")
        
        try:
            # Ensure proper shape for search
            query_vector = query_vector.astype(np.float32)
            
            # Perform search
            distances, indices = self.index.search(query_vector, k)
            return distances, indices
            
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index
        """
        logger.info(f"Saving index to {filepath}")
        try:
            faiss.write_index(self.index, filepath)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load the index from disk.
        
        Args:
            filepath: Path to load the index from
        """
        logger.info(f"Loading index from {filepath}")
        try:
            self.index = faiss.read_index(filepath)
            self.dimension = self.index.d
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise