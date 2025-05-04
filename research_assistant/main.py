#!/usr/bin/env python3

import argparse
import logging

from research_assistant.rag.assistant import ResearchAssistant
from paper_scraper.utils.logging_config import setup_logging

# Configure logging
logger = setup_logging()

def main():
    """Main entry point for the Research Assistant CLI."""
    parser = argparse.ArgumentParser(description="Research Assistant - RAG-based scientific literature QA")
    
    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Search papers command
    search_parser = subparsers.add_parser("search", help="Search and download scientific papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max-results", "-n", type=int, default=10,
                             help="Maximum number of papers to process (default: 10)")
    
    # Ask question command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the papers")
    ask_parser.add_argument("question", help="Question to answer")
    ask_parser.add_argument("--top-k", "-k", type=int, default=5,
                          help="Number of relevant chunks to use for answering (default: 5)")
    
    # Global options
    parser.add_argument("--model", "-m", default="mistralai/Mistral-7B-Instruct-v0.1",
                      help="Hugging Face model ID for the language model")
    parser.add_argument("--embedding-model", "-e", default="BAAI/bge-large-en-v1.5",
                      help="Embedding model to use for text vectorization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize assistant
    logger.info("Initializing Research Assistant...")
    assistant = ResearchAssistant(
        model_name=args.model,
        embedding_model_name=args.embedding_model
    )
    
    # Execute command
    if args.command == "search":
        logger.info(f"Searching for papers: {args.query}")
        assistant.search_papers(args.query, args.max_results)
    
    elif args.command == "ask":
        logger.info(f"Answering question: {args.question}")
        answer = assistant.answer_question(args.question, args.top_k)
        print(f"\nQuestion: {args.question}\n")
        print(f"Answer: {answer}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()