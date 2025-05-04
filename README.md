# Research Assistant with RAG

A modular system for scientific literature search and question-answering using Retrieval Augmented Generation (RAG).

## Overview

This project combines advanced natural language processing with scientific paper retrieval to create an AI research assistant. The system can search academic databases, process papers, and answer questions based on the retrieved information.

### Key Features

- **Academic Paper Retrieval**: Search and download papers from PubMed with full-text PDF support
- **Text Processing**: Extract and chunk text from scientific PDFs
- **Semantic Search**: Index paper content using state-of-the-art embeddings for quick retrieval
- **Question Answering**: Generate contextual answers to research questions using advanced language models
- **Modular Architecture**: Easily extendable components for different data sources or models

## Architecture

The project is organized into two main modules:

### 1. Paper Scraper Module
Handles the search and retrieval of scientific papers from academic databases.

- **Scraper**: Core functionality for interacting with publication APIs
- **Parsers**: Database-specific parsing logic (currently PubMed)
- **Downloaders**: PDF download and text extraction tools
- **Utils**: Helper functions and utilities

### 2. RAG Module
Implements the retrieval-augmented generation system for question answering.

- **Assistant**: Main integration class that coordinates all components
- **Embeddings**: Text vectorization using neural models
- **LLM**: Language model management for text generation
- **Indexing**: Vector storage and similarity search
- **Utils**: Text processing and helper functions

## Installation

### Setting Up a Virtual Environment

```bash
# Clone the repository
git clone https://github.com/zysea23/Med-RAG-assistant.git
cd research-assistant

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
## On Windows
venv\Scripts\activate
## On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### GPU Requirements

For optimal performance:
- CUDA-compatible GPU with 8GB+ VRAM
- CUDA Toolkit installed (version 11.7+ recommended)
- GPU drivers updated to latest version

## Usage

### Command Line Interface

```bash
# Make sure your virtual environment is activated
## On Windows
venv\Scripts\activate
## On macOS/Linux
source venv/bin/activate

# Search for papers on a topic
python -m research_assistant.main search "CRISPR gene editing cancer therapy" --max-results 10

# Ask a question about the papers
python -m research_assistant.main ask "What are the main challenges in using CRISPR for cancer therapy?"
```

### Python API

```python
# Make sure your virtual environment is activated before running Python

from research_assistant.rag.assistant import ResearchAssistant

# Initialize the assistant
assistant = ResearchAssistant()

# Search for papers
assistant.search_papers(
    query="latest developments in CRISPR gene editing",
    max_results=10
)

# Ask a question
answer = assistant.answer_question(
    "What are the main technical challenges in CRISPR delivery systems?",
    k=5  # Use top 5 most relevant chunks
)
print(answer)
```

## Configuration

The system components can be configured through environment variables:

```bash
# Set environment variables (before running the assistant)
## On Windows
set UNPAYWALL_EMAIL=your.email@domain.com
set HF_TOKEN=your_huggingface_token

## On macOS/Linux
export UNPAYWALL_EMAIL=your.email@domain.com
export HF_TOKEN=your_huggingface_token
```

Available environment variables:
- `UNPAYWALL_EMAIL`: Email for Unpaywall API access
- `CORE_API_KEY`: API key for CORE academic database (optional)
- `HF_TOKEN`: Hugging Face token for private model access (optional)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ VRAM for optimal LLM performance

## Extending the System

### Adding New Data Sources

Create a new parser in the `paper_scraper/parsers/` directory following the pattern of existing parsers.

### Using Different Models

Models can be specified when initializing the assistant:

```python
assistant = ResearchAssistant(
    model_name="meta-llama/Llama-2-13b-chat-hf",  # Different LLM
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"  # Different embeddings
)
```

## Troubleshooting

### Common Issues

- **Out of VRAM**: Try using a smaller model or enabling model quantization
- **PDF Download Failures**: Check network connectivity and try again later
- **Slow Embedding Generation**: Reduce batch size in embedding configuration