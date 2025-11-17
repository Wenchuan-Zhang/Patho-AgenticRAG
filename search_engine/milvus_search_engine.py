import os
import time
from pathlib import Path

from scipy.stats import kurtosis

from code_discovery.time_util import timeit

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from pymilvus import MilvusClient
import base64
from PIL import Image
from io import BytesIO
import torch
from search_engine.milvus import MilvusColQwen
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
from colpali_engine.models import ColQwen2, ColQwen2Processor
import json
from typing import List, Any
import numpy as np


def exist(pdf_path, pdf_image_page, paths):
    """Check if the target document exists in the search results"""
    for path in paths:
        if pdf_path in path and (f"_{pdf_image_page}" in path):
            return True
    return False


class MilvusSearchEngine:
    def __init__(
        self,
        milvus_uri,
        collection_name="pdf_datasets",
        embed_model_name="vidore/colqwen2-v1.0",
        device_str="cpu",
        cache_dir="cache",
    ):
        self.collection_name = collection_name
        self.client = MilvusClient(uri=milvus_uri)
        self.milvus = MilvusColQwen(self.client, collection_name, dim=128)
        print("Device", device_str)
        self.device = get_torch_device(device_str)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.vector_embed_model = ColQwen2.from_pretrained(
            embed_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

        self.processor = ColQwen2Processor.from_pretrained(embed_model_name)

    def save_search_cache(
        self,
        entry_id: int,
        text_results: List,
        text_embed: np.ndarray,
        img_embed: np.ndarray,
    ):
        """
        Save text_results and embeddings (separately for text and image) to disk for a specific entry.

        Args:
            entry_id: Unique identifier for the dataset entry
            text_results: List of search results from text search
            text_embed: Text embeddings
            img_embed: Image embeddings
        """
        entry_dir = self.cache_dir / f"entry_{entry_id}"
        entry_dir.mkdir(exist_ok=True)

        # Save text_results as JSON (convert numpy arrays to lists for JSON serialization)
        text_results_serializable = []
        for result in text_results:
            result_dict = {
                "score": float(result[0]) if len(result) > 0 else None,
                "id": result[1] if len(result) > 1 else None,
                "filepath": result[2] if len(result) > 2 else None,
                "embedding": result[3].tolist()
                if len(result) > 3 and hasattr(result[3], "tolist")
                else None,
            }
            text_results_serializable.append(result_dict)

        with open(entry_dir / "text_results.json", "w") as f:
            json.dump(text_results_serializable, f, indent=2)

        # Save text and image embeddings separately
        np.save(entry_dir / "text_embed.npy", text_embed)
        np.save(entry_dir / "img_embed.npy", img_embed)

        print(f"Cached results for entry {entry_id} saved to {entry_dir}")

    def load_search_cache(self, entry_id: int):
        """
        Load cached text_results, text_embed, and img_embed for a specific entry.

        Args:
            entry_id: Unique identifier for the dataset entry

        Returns:
            tuple: (text_results, text_embed, img_embed) or (None, None, None) if cache doesn't exist
        """
        entry_dir = self.cache_dir / f"entry_{entry_id}"
        text_results_file = entry_dir / "text_results.json"
        text_embed_file = entry_dir / "text_embed.npy"
        img_embed_file = entry_dir / "img_embed.npy"

        if not (
            text_results_file.exists()
            and text_embed_file.exists()
            and img_embed_file.exists()
        ):
            return None, None, None

        # Load text_results
        with open(text_results_file, "r") as f:
            text_results_data = json.load(f)

        # Convert back to original format
        text_results = []
        for result_dict in text_results_data:
            result = [
                result_dict.get("score"),
                result_dict.get("id"),
                result_dict.get("filepath"),
                np.array(result_dict["embedding"])
                if result_dict.get("embedding")
                else None,
            ]
            text_results.append(result)

        # Load text_embed and img_embed
        text_embed = np.load(text_embed_file)
        img_embed = np.load(img_embed_file)

        print(f"Loaded cached results for entry {entry_id}")
        return text_results, text_embed, img_embed

    def cache_exists(self, entry_id: int) -> bool:
        """Check if cache exists for a specific entry."""
        entry_dir = self.cache_dir / f"entry_{entry_id}"
        return (
            (entry_dir / "text_results.json").exists()
            and (entry_dir / "text_embed.npy").exists()
            and (entry_dir / "img_embed.npy").exists()
        )

    @timeit
    def batch_search(
        self,
        queries: List[str],
        images: List[Any],
        topk=10,
        partition=None,
        entry_id=None,
        use_cache=True,
    ):
        start_time = time.time()
        """Enhanced batch search with caching support, saving text and image embeddings separately."""
        # Try to load from cache first
        if entry_id is not None and use_cache and self.cache_exists(entry_id):
            book_search_results, text_embed, img_embed = self.load_search_cache(
                entry_id
            )
            return self._rerank_and_return_results(
                book_search_results, text_embed, img_embed, topk
            )

        # Original search logic
        book_search_results = []
        img_embeds = []

        if queries:
            dataloader = DataLoader(
                dataset=ListDataset[str](queries),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_queries(x),
            )

            qs: List[torch.Tensor] = []
            for batch_query in dataloader:
                with torch.no_grad():
                    batch_query = {
                        k: v.to(self.vector_embed_model.device)
                        for k, v in batch_query.items()
                    }
                    embeddings_query = self.get_text_embedding(batch_query)
                    qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

            for query in qs:
                query = query.float().numpy()
                retrieval_topk = 20
                if partition:
                    search_results = self.milvus.search(
                        query, topk=retrieval_topk, partition=[partition]
                    )
                else:
                    search_results = self.milvus.search(query, topk=retrieval_topk)
                book_search_results.extend(search_results)

        # Image query embedding
        if images is not None:
            for img in images:
                if img is None:
                    continue
                if isinstance(img, str):
                    if img.startswith("data:image"):
                        img_data = base64.b64decode(img.split(",")[-1])
                        img = Image.open(BytesIO(img_data))
                    else:
                        img = Image.open(img)
                elif isinstance(img, dict) and "bytes" in img:
                    img = Image.open(BytesIO(img["bytes"]))

                img_embed_single = self.get_image_embedding(img)
                img_embed_single = img_embed_single[0]
                img_embeds.append(img_embed_single.squeeze(0).float().cpu().numpy())

        final_results = []
        # Combine the embeddings
        for text_embed, img_embed in zip(qs, img_embeds):
            text_embed = text_embed.float().cpu().numpy()

            # Save to cache if entry_id is provided
            if entry_id is not None:
                self.save_search_cache(
                    entry_id, book_search_results, text_embed, img_embed
                )

            print(f"Batch search took {time.time() - start_time:.4f} seconds")
            top_filepaths, all_filepaths = self._rerank_and_return_results(
                book_search_results, text_embed, img_embed, topk
            )
            if entry_id is not None:
                # for offline evaluation
                assert len(img_embeds) == 1
                return top_filepaths, all_filepaths
            else:
                # only return top-1
                final_results.append([top_filepaths[0]])
        return final_results

    @timeit
    def get_text_embedding(self, batch_query):
        embeddings_query = self.vector_embed_model(**batch_query)
        return embeddings_query

    @timeit
    def get_image_embedding(self, img):
        processed = self.processor.process_images([img]).to(
            self.vector_embed_model.device
        )
        with torch.no_grad():
            img_embed_single = self.vector_embed_model(**processed)
        return img_embed_single

    def _rerank_and_return_results(self, text_results, text_embed, img_embed, topk):
        start_time = time.time()
        """Perform reranking using cached or fresh text_results, text_embed, and img_embed."""
        top_results = []
        for res in text_results:
            db_embed = res[3]

            # Compute cosine sim between each row of combined_embed and db_embed
            # concat_embed = np.concatenate([text_embed, img_embed], axis=0)
            # sims = (
            #     concat_embed @ db_embed.T
            # )  # Assuming we are reranking with text embedding first
            # max_sim = np.mean(sims.max(1))

            print("Rerank!")
            text_sim = text_embed @ db_embed.T
            img_sim = img_embed @ db_embed.T
            max_sim = np.std(np.std(text_sim, axis=1)) * (
                np.mean(kurtosis(text_sim, axis=1)) ** 2
            ) * np.mean(kurtosis(img_sim, axis=1)) + np.mean(np.max(text_sim, axis=1))

            similarity = max_sim
            filepath = res[2]
            top_results.append((similarity, filepath))

        # Sort and return top_k
        top_results.sort(key=lambda x: x[0], reverse=True)
        all_filepaths = [r[1] for r in top_results]
        final_top_results = top_results[:topk]
        top_filepaths = [r[1] for r in final_top_results]
        print(f"Reranking took {time.time() - start_time:.4f} seconds")
        print(f"Best score (combined): {top_filepaths}")
        return top_filepaths, all_filepaths

    def rerank_from_cache(self, entry_id: int, topk=10, custom_weights=None):
        start_time = time.time()
        """Perform reranking using only cached data with optional custom weights."""
        text_results, combined_embed = self.load_search_cache(entry_id)
        if text_results is None:
            raise ValueError(f"No cache found for entry {entry_id}")

        # Apply custom reranking logic here if needed
        if custom_weights is not None:
            # Example: apply custom weights to combined_embed
            combined_embed = combined_embed * custom_weights

        print(f"Rerank from cache took {time.time() - start_time:.4f} seconds")
        return self._rerank_and_return_results(text_results, combined_embed, topk)

    def clear_cache(self, entry_id=None):
        """Clear cache for specific entry or all entries."""
        if entry_id is not None:
            entry_dir = self.cache_dir / f"entry_{entry_id}"
            if entry_dir.exists():
                import shutil

                shutil.rmtree(entry_dir)
                print(f"Cleared cache for entry {entry_id}")
        else:
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir()
                print("Cleared all cache")


def load_dataset(json_path: str):
    with open(json_path, "r") as file:
        dataset = json.load(file)
    return dataset


def process_and_search(engine, dataset, topk, use_cache=True, force_refresh=False):
    start_time = time.time()
    score = 0
    top_100_score = 0
    processed = 0

    for i, entry in enumerate(dataset):
        entry_id = entry.get("No", i)  # Use 'No' field or index as entry_id
        question = entry["question"]
        image_path = "rag_dataset/" + entry["img"]
        # options = entry["options"]
        # prompt = question + "\n" + "\n".join(options)

        options = entry["answer"]
        prompt = question + "\n" + options
        print(f"Searching for entry {entry_id}: {prompt}")

        # Clear cache for this entry if force_refresh is True
        if force_refresh:
            engine.clear_cache(entry_id)

        # Perform the search with caching
        top_results, all_results = engine.batch_search(
            [prompt],
            images=[image_path],
            topk=topk,
            partition="General_Comprehensive",
            entry_id=entry_id,
            use_cache=use_cache,
        )

        print(f"Top search results for question {entry_id}: {top_results}")

        if exist(entry["pdf_path"], entry["pdf_image_page"], top_results):
            score += 1
        if exist(entry["pdf_path"], entry["pdf_image_page"], all_results):
            top_100_score += 1
        processed += 1
        print("Recall:", score / processed, "Full Recall:", top_100_score / processed)

    print(f"Total process took {time.time() - start_time:.4f} seconds")


