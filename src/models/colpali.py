"""ColPali model wrapper for vision-language understanding."""

import gc
import os
from typing import Optional, Union

import numpy as np
import torch
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColIdefics3,
    ColIdefics3Processor,
)
from PIL import Image

from ..utils.timer import Timer


class ColPaliModel:
    """Wrapper for ColPali and related vision-language models."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """Initialize the ColPali model.

        Args:
            pretrained_model_name_or_path: Model name or path
            model_dtype: Model data type
            device: Device to run the model on
            **kwargs: Additional arguments
        """
        if (
            "colpali" not in pretrained_model_name_or_path.lower()
            and "colqwen2" not in pretrained_model_name_or_path.lower()
            and "colsmol" not in pretrained_model_name_or_path.lower()
        ):
            raise ValueError(
                "This module only supports ColPali, ColQwen2, ColQwen2.5, ColSmol-256M & ColSmol-500M for now. Incorrect model name specified."
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_dtype = model_dtype
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if "colpali" in pretrained_model_name_or_path.lower():
            self.model = ColPali.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2.5" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2_5.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colsmol" in pretrained_model_name_or_path.lower():
            self.model = ColIdefics3.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )

        # set to eval mode
        self.model = self.model.eval()

        self.patches = tuple()

        if "colpali" in pretrained_model_name_or_path.lower():
            self.processor = ColPaliProcessor.from_pretrained(
                self.pretrained_model_name_or_path,
                use_fast=True,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2.5" in pretrained_model_name_or_path.lower():
            self.processor = ColQwen2_5_Processor.from_pretrained(
                self.pretrained_model_name_or_path,
                use_fast=True,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.processor = ColQwen2Processor.from_pretrained(
                self.pretrained_model_name_or_path,
                use_fast=True,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colsmol" in pretrained_model_name_or_path.lower():
            self.processor = ColIdefics3Processor.from_pretrained(
                self.pretrained_model_name_or_path,
                use_fast=True,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """Create a ColPaliModel from a pretrained model."""
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            **kwargs,
        )

    def encode_queries(self, texts: Union[str, list[str]]) -> np.ndarray:
        """Encode text queries into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        with torch.inference_mode():
            batch_text = self.processor.process_texts(texts).to(self.model.device)
            embeddings = self.model(**batch_text).detach().cpu().float().numpy()
        del batch_text
        torch.cuda.empty_cache()
        return embeddings

    def encode_images(
        self, images: Union[Image.Image, list[Image.Image]]
    ) -> np.ndarray:
        """Encode images into embeddings."""
        if isinstance(images, Image.Image):
            images = [images]
        with torch.inference_mode():
            batch_images = self.processor.process_images(images).to(self.model.device)
            embeddings = self.model(**batch_images).detach().cpu().float().numpy()
        del batch_images
        torch.cuda.empty_cache()
        return embeddings

    def get_patches(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Get the number of patches for an image size."""
        if "colpali" in self.pretrained_model_name_or_path.lower():
            return self.processor.get_n_patches(image_size, patch_size=self.model.patch_size)  # type: ignore
        elif "colqwen" in self.pretrained_model_name_or_path.lower():
            return self.processor.get_n_patches(image_size, spatial_merge_size=self.model.spatial_merge_size)  # type: ignore
        return 0, 0

    @Timer("Batched Pooled Embeddings")
    def batch_pooled_embeddings(self, image_batch: list[Image.Image]):
        """Generate batched pooled embeddings for images."""
        # embed
        with torch.inference_mode():
            processed_images = self.processor.process_images(image_batch).to(
                self.model.device
            )
            image_embeddings = self.model(**processed_images)

        tokenized_images = processed_images.input_ids

        del processed_images
        torch.cuda.empty_cache()

        # mean pooling
        pooled_by_rows_batch = []
        pooled_by_columns_batch = []

        for image_embedding, tokenized_image, image in zip(
            image_embeddings, tokenized_images, image_batch
        ):
            x_patches, y_patches = self.get_patches(image.size)

            image_tokens_mask = tokenized_image == self.processor.image_token_id

            # Number of actual image tokens for this sample
            num_image_tokens = image_tokens_mask.sum().item()
            # Derive x/y patch counts safely
            if (
                x_patches <= 0
                or y_patches <= 0
                or (x_patches * y_patches != num_image_tokens)
            ):
                # Default to square-ish layout if metadata is wrong
                y_patches = int((num_image_tokens) ** 0.5)
                x_patches = num_image_tokens // y_patches

            image_tokens = image_embedding[image_tokens_mask].view(
                x_patches, y_patches, self.model.dim
            )
            pooled_by_rows = torch.mean(image_tokens, dim=0)
            pooled_by_columns = torch.mean(image_tokens, dim=1)

            image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
            first_image_token_idx = image_token_idxs[0].cpu().item()
            last_image_token_idx = image_token_idxs[-1].cpu().item()

            if first_image_token_idx == 0 and last_image_token_idx == len(
                image_embedding - 1
            ):
                pooled_by_rows = pooled_by_rows.cpu().float().numpy().tolist()
                pooled_by_columns = pooled_by_columns.cpu().float().numpy().tolist()
            else:
                prefix_tokens = image_embedding[:first_image_token_idx]
                postfix_tokens = image_embedding[last_image_token_idx + 1 :]

                # adding back prefix and postfix special tokens
                pooled_by_rows = (
                    torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0)
                    .cpu()
                    .float()
                    .numpy()
                    .tolist()
                )
                pooled_by_columns = (
                    torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0)
                    .cpu()
                    .float()
                    .numpy()
                    .tolist()
                )

            pooled_by_rows_batch.append(pooled_by_rows)
            pooled_by_columns_batch.append(pooled_by_columns)

            del (
                x_patches,
                y_patches,
                image_tokens_mask,
                image_tokens,
                pooled_by_rows,
                pooled_by_columns,
                first_image_token_idx,
                last_image_token_idx,
                prefix_tokens,
                postfix_tokens,
            )

        image_embeddings = image_embeddings.cpu().float().numpy().tolist()

        del tokenized_images
        torch.cuda.empty_cache()
        gc.collect()

        return image_embeddings, pooled_by_rows_batch, pooled_by_columns_batch
