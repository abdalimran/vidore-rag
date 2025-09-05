"""Main script demonstrating Vidore RAG usage."""

import os
from pathlib import Path

from dotenv import load_dotenv

from vidore_rag import RAG

# Load environment variables
load_dotenv(".env")


def main():
    """Main function demonstrating RAG usage."""
    # Initialize the RAG system
    rag = RAG("vidore/colpali-v1.3")

    # Example: Index a PDF file
    # rag.index_file(pdf_path=Path("attention_is_all_you_need.pdf"), batch_size=1)

    # Example: Answer a question
    response = rag.answer(
        query="How does multi headed attention work?",
        top_k=4,
        prefetch_limit=10,
    )

    print(f"Response: {response}")

    # Clean up
    rag.close()


if __name__ == "__main__":
    main()
