#!/usr/bin/env python3
"""
Example usage of the Research Assistant.
"""

import logging
import torch
from huggingface_hub import login
from research_assistant.rag.assistant import ResearchAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Check GPU availability
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Authenticate with Hugging Face (if needed)
    # Replace with your token - better to use environment variables in practice
    # login("your_huggingface_token")
    
    # Initialize the assistant
    logger.info("Initializing Research Assistant...")
    assistant = ResearchAssistant(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        embedding_model_name="BAAI/bge-large-en-v1.5",
        output_dir="research_output"
    )
    
    # Search for papers on a topic
    logger.info("Searching for papers...")
    assistant.search_papers(
        query="latest developments in CRISPR gene editing cancer therapy",
        max_results=5  # Limit for demonstration
    )
    
    # Ask questions about the papers
    questions = [
        "What are the main challenges in using CRISPR for cancer therapy?",
        "What are some recent breakthroughs in CRISPR cancer treatments?"
    ]
    
    for question in questions:
        logger.info(f"Question: {question}")
        answer = assistant.answer_question(question)
        print(f"\nQ: {question}")
        print(f"\nA: {answer}\n{'-'*80}")

if __name__ == "__main__":
    main()