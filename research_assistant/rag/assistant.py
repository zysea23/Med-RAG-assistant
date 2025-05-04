"""
Research Assistant class implementing RAG for scientific literature QA.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import json
from datetime import datetime

# Import components
from paper_scraper.scraper import PaperScraper
from research_assistant.rag.embeddings.encoder import EmbeddingModel
from research_assistant.rag.llm.model import LanguageModel
from research_assistant.rag.indexing.vector_store import VectorIndex
from research_assistant.rag.utils.text_processing import chunk_text, clean_text

logger = logging.getLogger(__name__)

class ResearchAssistant:
    """
    Research Assistant for scientific literature search and question answering.
    
    This class combines paper retrieval, embedding, vector search, and language
    generation to create a RAG-based research assistant.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        output_dir: str = "research_output"
    ):
        """
        Initialize the Research Assistant.
        
        Args:
            model_name: Hugging Face model ID for the language model
            embedding_model_name: Model for text embeddings
            output_dir: Directory to store output files
        """
        logger.info("Initializing Research Assistant components")
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scraper = PaperScraper(output_dir=str(self.output_dir / "papers"))
        
        # Initialize embedding model
        logger.info("Loading embedding model")
        self.embedding_model = EmbeddingModel(model_name=embedding_model_name)
        
        # Initialize language model
        logger.info("Loading language model")
        self.language_model = LanguageModel(model_name=model_name)
        
        # Initialize vector index with embedding dimension
        embedding_dim = self.embedding_model.embedding_dim
        logger.info(f"Creating vector index with dimension {embedding_dim}")
        self.vector_index = None
        
        # Storage for paper contents and metadata
        self.paper_texts: List[str] = []
        self.paper_metadata: List[Dict] = []
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        
        logger.info("Research Assistant initialized successfully")
    
    def search_papers(self, query: str, max_results: int = 10) -> None:
        """
        Search for papers on the given topic and download their content.
        
        Args:
            query: Search query for scientific papers
            max_results: Maximum number of papers to retrieve
        """
        logger.info(f"Searching for papers about: {query}")
        
        # Search for paper IDs
        pmids = self.scraper.search_pubmed(query, max_results)
        
        # Get detailed information
        papers = self.scraper.fetch_pubmed_details(pmids)
        
        # Reset existing storage
        self.paper_texts = []
        self.paper_metadata = []
        
        # Process each paper
        for paper in papers:
            pdf_url = paper.get('full_text_link')
            if not pdf_url:
                logger.warning(f"No PDF available for paper {paper.get('pubmed_id')}")
                continue
                
            try:
                # Create a safe filename from the title
                title = paper.get('title', '')
                safe_title = ''.join(c.lower() for c in title if c.isalnum() or c.isspace())
                safe_title = safe_title.replace(' ', '_')[:100]
                
                # Fall back to PMID if title processing results in empty string
                filename = f"{safe_title or paper['pubmed_id']}.pdf"
                pdf_path = self.output_dir / "papers" / "pdf" / filename
                
                # Download PDF
                self.scraper.download_pdf(pdf_url, str(pdf_path))
                
                # Extract text from PDF
                text = self.scraper.extract_text_from_pdf(str(pdf_path))
                
                # Clean text
                text = clean_text(text)
                
                # Store paper content and metadata
                if text:
                    self.paper_texts.append(text)
                    self.paper_metadata.append(paper)
                    logger.info(f"Successfully processed paper: {title}")
                else:
                    logger.warning(f"Failed to extract text from paper: {title}")
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('pubmed_id', 'unknown')}: {e}")
        
        logger.info(f"Successfully processed {len(self.paper_texts)} papers")
        
        # Create search index from papers
        self._build_index()
        
        # Save metadata of successfully processed papers
        self._save_paper_metadata()
    
    def _build_index(self) -> None:
        """
        Process papers into chunks and build search index.
        """
        logger.info("Building search index from papers")
        
        # Reset chunks
        self.chunks = []
        self.chunk_metadata = []
        
        # Process each paper into chunks
        for text, metadata in zip(self.paper_texts, self.paper_metadata):
            # Split text into chunks
            paper_chunks = chunk_text(
                text, 
                chunk_size=500,  # Words per chunk
                chunk_overlap=50  # Words of overlap
            )
            
            # Add each chunk with its metadata
            for chunk in paper_chunks:
                if len(chunk.split()) >= 30:  # Only keep substantive chunks
                    self.chunks.append(chunk)
                    self.chunk_metadata.append(metadata)
        
        logger.info(f"Created {len(self.chunks)} chunks from {len(self.paper_texts)} papers")
        
        # Generate embeddings for all chunks
        if self.chunks:
            embeddings = self.embedding_model.encode(
                self.chunks,
                batch_size=8,
                show_progress=True
            )
            
            # Initialize vector index with proper dimension
            embedding_dim = embeddings.shape[1]
            self.vector_index = VectorIndex(dimension=embedding_dim)
            
            # Add embeddings to index
            self.vector_index.add_vectors(embeddings)
            logger.info(f"Added {len(embeddings)} vectors to search index")
        else:
            logger.warning("No chunks to index")
    
    def _save_paper_metadata(self) -> None:
        """
        Save metadata of processed papers.
        """
        if not self.paper_metadata:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = self.output_dir / "papers" / "metadata"
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_path / f"papers_{timestamp}.json"
        
        with open(metadata_file, "w") as f:
            json.dump(self.paper_metadata, f, indent=2)
            
        logger.info(f"Saved metadata to {metadata_file}")
    
    def answer_question(self, question: str, k: int = 5) -> str:
        """
        Answer a question using RAG with the processed papers.
        
        Args:
            question: User's question about the papers
            k: Number of most relevant chunks to use
            
        Returns:
            Generated answer with citations
        """
        logger.info(f"Answering question: {question}")
        
        if not self.chunks or not self.vector_index:
            logger.error("No papers indexed. Search for papers first.")
            return "Please search for papers first before asking questions."
        
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode(question)
        
        # Find most similar chunks
        distances, indices = self.vector_index.search(
            question_embedding.reshape(1, -1), k=k
        )
        
        # Build context from relevant chunks
        context = ""
        used_papers = set()
        total_tokens = 0
        max_tokens = 2048  # Limit total context length
        
        for idx in indices[0]:
            chunk = self.chunks[idx]
            metadata = self.chunk_metadata[idx]
            paper_id = metadata['pubmed_id']
            
            # Only include first chunk from each paper and check token length
            if paper_id not in used_papers:
                # Estimate token count (rough approximation)
                chunk_tokens = len(chunk.split()) * 1.3
                if total_tokens + chunk_tokens > max_tokens:
                    break
                    
                # Add paper citation and chunk to context
                used_papers.add(paper_id)
                paper_citation = f"'{metadata['title']}' ({metadata.get('journal', {}).get('name', 'Journal')})"
                context += f"\n\nFrom paper {paper_citation}:\n{chunk}\n"
                total_tokens += chunk_tokens
        
        # Construct prompt for the language model
        prompt = f"""Answer the following question based on these research paper excerpts. 
        Include citations to the papers when referencing specific information.

        Research paper excerpts:
        {context}

        Question: {question}

        Answer: """
        
        # Generate answer using language model
        answer = self.language_model.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        logger.info("Generated answer successfully")
        return answer