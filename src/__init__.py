"""Vidore RAG - A Retrieval-Augmented Generation system for document analysis."""

__version__ = "0.1.0"

from .utils.timer import Timer
from .models.colpali import ColPaliModel
from .vectorstore.qdrant_store import QdrantVectorStore
from .rag.system import RAG

__all__ = ["Timer", "ColPaliModel", "QdrantVectorStore", "RAG"]
