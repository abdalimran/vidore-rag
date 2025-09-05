# Vidore RAG

A Retrieval-Augmented Generation (RAG) system for document analysis using vision-language models.

## Features

- **Document Indexing**: Convert PDF documents to images and index them using ColPali models
- **Vector Search**: Efficient similarity search using Qdrant vector database
- **Question Answering**: Answer questions based on document content using LLM
- **Streamlit Web App**: User-friendly web interface for document upload and Q&A
- **Modular Design**: Clean, organized code structure following Python best practices

## Installation

```bash
pip install -e .
```

## Quick Start

### Using the Streamlit Web App

```bash
# Install dependencies
pip install -e .

# Run the Streamlit app
streamlit run streamlit_app.py
```

Then open your browser to the provided URL and start uploading PDFs and asking questions!

### Using the Python API

```python
from vidore_rag import RAG

# Initialize the RAG system
rag = RAG("vidore/colpali-v1.3")

# Index a PDF file
rag.index_file("path/to/your/document.pdf")

# Ask questions
response = rag.answer("What is the main topic of the document?")
print(response)

# Clean up
rag.close()
```

## Project Structure

```
vidore_rag/
├── __init__.py
├── utils/
│   ├── __init__.py
│   └── timer.py          # Timer utility for performance monitoring
├── models/
│   ├── __init__.py
│   └── colpali.py        # ColPali model wrapper
├── vectorstore/
│   ├── __init__.py
│   └── qdrant_store.py   # Qdrant vector store implementation
└── rag/
    ├── __init__.py
    └── system.py         # Main RAG system orchestration
streamlit_app.py           # Streamlit web application
main.py                    # Command-line interface
```

## Components

### Timer (`vidore_rag.utils.timer`)
A versatile timer utility that can be used as a decorator or context manager for performance monitoring.

### ColPaliModel (`vidore_rag.models.colpali`)
Wrapper for ColPali and related vision-language models (ColQwen2, ColSmol) with methods for:
- Text query encoding
- Image encoding
- Batch processing with pooling

### QdrantVectorStore (`vidore_rag.vectorstore.qdrant_store`)
Qdrant-based vector store implementation with:
- Multi-vector support
- Batch uploads
- Similarity search with prefetching

### RAG System (`vidore_rag.rag.system`)
Main orchestration class that combines:
- PDF processing and image conversion
- Document indexing
- Question answering with LLM integration

## Configuration

Set environment variables for LLM configuration:

```bash
export API_KEY="your-api-key"
export BASE_URL="http://localhost:11434"
export MODEL_NAME="gemma3:4b"
```

Or create a `.env` file:

```
API_KEY=your-api-key
BASE_URL=http://localhost:11434
MODEL_NAME=gemma3:4b
```

## Usage Examples

### Index Multiple PDFs

```python
from vidore_rag import RAG

rag = RAG()
rag.index_folder("path/to/pdf/folder", batch_size=4)
```

### Custom Model Configuration

```python
from vidore_rag import RAG
import torch

rag = RAG(
    rag_model="vidore/colSmol-500M",
    model_dtype=torch.float16,
    # device will be auto-detected (cuda/mps/cpu)
    index_path="./custom_index",
    collection_name="my_docs"
)
```

## Dependencies

- `colpali-engine`: Vision-language models
- `qdrant-client`: Vector database
- `litellm`: LLM API integration
- `torch`: Deep learning framework
- `pillow`: Image processing
- `pdf2image`: PDF to image conversion
- `numpy`: Numerical computing
- `streamlit`: Web application framework

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request
