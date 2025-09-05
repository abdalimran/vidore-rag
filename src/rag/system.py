"""RAG system for document analysis and question answering."""

import base64
import io
import os
from pathlib import Path
from textwrap import dedent
from typing import Optional, Union

import litellm
import numpy as np
import torch
from pdf2image import convert_from_path
from PIL import Image

from ..models.colpali import ColPaliModel
from ..utils.timer import Timer
from ..vectorstore.qdrant_store import QdrantVectorStore


class RAG:
    """Retrieval-Augmented Generation system for document analysis."""

    def __init__(
        self,
        rag_model: str = "vidore/colqwen2-v1.0",
        model_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        index_path: str = "./vector_store",
        collection_name: str = "rag",
    ) -> None:
        """Initialize the RAG system.

        Args:
            rag_model: Name of the RAG model
            model_dtype: Model data type
            device: Device to run the model on
            index_path: Path to the vector store index
            collection_name: Name of the collection
        """
        self.model = ColPaliModel(
            pretrained_model_name_or_path=rag_model,
            model_dtype=model_dtype,  # type: ignore
            device=device,
        )
        self.vector_store = QdrantVectorStore(
            index_path=index_path, collection_name=collection_name
        )

    @Timer("Index Image Batch")
    def _index_batch(self, image_queue: list):
        """Index a batch of images."""
        image_batch = []
        payload_batch = []
        while image_queue:
            image, payload = image_queue.pop(0)
            image_batch.append(image)
            payload_batch.append(payload)

        original_batch, pooled_by_rows_batch, pooled_by_columns_batch = (
            self.model.batch_pooled_embeddings(image_batch)
        )
        self.vector_store.upload_batch_vectors(
            np.asarray(original_batch),
            np.asarray(pooled_by_rows_batch),
            np.asarray(pooled_by_columns_batch),
            [payload],
        )
        del original_batch, pooled_by_rows_batch, pooled_by_columns_batch

    @Timer("PDF to Image")
    def _pdf_to_image(self, pdf_path: Union[str, Path]):
        """Convert PDF pages to images."""
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path).resolve()

        images = convert_from_path(
            pdf_path=str(pdf_path),
            dpi=100,
            thread_count=max(1, int(os.cpu_count() // 2)),  # type: ignore
        )

        for page_no, image in enumerate(images, 1):
            buffer = io.BytesIO()
            image.save(buffer, format="jpeg", quality=75)
            yield image, {
                "file": pdf_path.name,
                "page_no": page_no,
                "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            }
            del buffer, image

    @Timer("Index Folder")
    def index_folder(self, path: Union[str, Path], batch_size: int = 1):
        """Index all PDFs in a folder."""
        if isinstance(path, str):
            path = Path(path).resolve()

        pdf_files = list(path.glob("*.pdf"))
        image_queue = []
        for pdf_path in pdf_files:
            self.index_file(pdf_path=pdf_path, batch_size=batch_size)

        del image_queue

    @Timer("Index PDF File")
    def index_file(self, pdf_path: Union[str, Path], batch_size: int = 1):
        """Index a single PDF file."""
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path).resolve()

        image_queue = []
        for image, payload in self._pdf_to_image(pdf_path):
            image_queue.append((image, payload))

            if len(image_queue) >= batch_size:
                self._index_batch(image_queue)

        del image_queue

    @Timer("Answer")
    def answer(
        self,
        query: str,
        top_k: int = 2,
        prefetch_limit: int = 10,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        with_images: bool = False,
    ):
        """Answer a question using the indexed documents."""
        if not api_key:
            api_key = os.getenv("API_KEY", "ollama")
        if not base_url:
            base_url = os.getenv("BASE_URL", "http://localhost:11434")
        if not model_name:
            model_name = os.getenv("MODEL_NAME", "gemma3:4b")

        queries = self.model.encode_queries(query)
        points = self.vector_store.search(
            queries=queries,
            search_limit=top_k,
            prefetch_limit=prefetch_limit,
        )[0]

        system_prompt = dedent(
            f"""
        You are an intelligent assistant that answers user questions only based on the provided context (the images).

        You are given pages from documents. Use its visual and textual information to accurately and concisely answer the user's question. If the answer is not present in the image, clearly state that the answer cannot be found.

        Follow these rules:
        - Base your response only on the image content unless explicitly instructed otherwise.
        - Do not hallucinate or make assumptions beyond the visible data.
        - If the image contains tables, diagrams, or equations, interpret them accurately.

        Your job is to provide an accurate and detailed answer to the user's question based on the provided context.
        """
        )

        llm_message: list = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        ]

        image_msgs: list = []
        images: list[dict] = []

        for point in points:
            encoded_image = point.payload["image"]  # type: ignore
            image_msgs.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            )
            images.append(point.payload)

        user_content = llm_message[1]["content"]
        llm_message[1]["content"] = user_content + image_msgs

        del points, image_msgs, queries

        print(f"Generating Answer...")
        response = (
            litellm.completion(
                model=f"ollama/{model_name}",
                messages=llm_message,
            )
            .choices[0]  # type: ignore
            .message.content  # type: ignore
        )
        print(f"Answer: {response}")

        if with_images:
            return response, images

        return response

    def close(self):
        """Cleanly close the Qdrant client to avoid shutdown-time ImportError."""
        try:
            if (
                hasattr(self.vector_store, "client")
                and self.vector_store.client is not None
            ):
                self.vector_store.client.close()
        except Exception as e:
            print(f"Error while closing Qdrant client: {e}")
