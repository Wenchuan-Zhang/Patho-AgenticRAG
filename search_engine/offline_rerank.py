"""
Offline multimodal re-ranking (Section 3.3 / Table 1 of the Patho-AgenticRAG paper).

Pipeline per query:
  1. Take the cached text-retrieval candidates of the query.
  2. Keep the top-20 pages by ColPali-style text late-interaction score
     mean_i max_j S_t[i, j].
  3. Re-rank these 20 pages with the multimodal fusion formula, Eq. (1).

The script works purely on cached data, so no Milvus server or GPU is needed.
Expected layout (paths are relative to the repository root):

  rag_dataset/dataset.json            # list of entries with keys: No, question, options,
                                      # pdf_path, pdf_image_page, ...
  cache/entry_<No>/text_results.json  # text-retrieval candidates:
                                      # [{"score", "id", "filepath", "embedding"}, ...]
  cache/entry_<No>/text_embed.npy     # query text token embeddings  (N_t, D)
  cache/entry_<No>/img_embed.npy      # query image patch embeddings (N_v, D)

Run from the repository root:
  python -m search_engine.offline_rerank

The dataset is split 50/50 (random_state=0); the fusion formula was searched on
the training half (see multimodal_fusion/) and is evaluated here on the test half.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from multimodal_fusion.basic_functions import *
import torch
from multimodal_fusion.basic_functions import USE_GPU

top_k = 1

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


def normalize(x):
    # return x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def exist(
    pdf_path="ROSAI and ACKERMAN'S SURGICAL PATHOLOGY 10th edition",
    pdf_image_page=1596,
    paths=None,
):
    if not paths:
        return False
    for path in paths:
        if pdf_path in path and (f"_{pdf_image_page}" in path):
            return True
    return False


def calculate_mean_reciprocal_rank(entry, top_results):
    for idx, item in enumerate(top_results):
        if exist(entry["pdf_path"], entry["pdf_image_page"], [item]):
            # If relevant item is found, return the reciprocal of the rank
            return 1 / (idx + 1)
    # If no relevant item is found, return 0
    return 0


class MilvusSearchEngine:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    # @timeit
    def load_search_cache(self, entry_id: int):
        """
        Load cached text_results, text_embeddings, and image_embeddings for a specific entry.
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

        # Load text and image embeddings separately
        text_embed = np.load(text_embed_file)
        img_embed = np.load(img_embed_file)

        # print(f"Loaded cached results for entry {entry_id}")
        return text_results, text_embed, img_embed

    def rerank_from_cache(self, entry_id: int, topk=10, fusion_similarity_fn=None):
        """
        Perform reranking using only cached data, fusion strategy, and similarity calculation.
        """
        text_results, text_embed, img_embed = self.load_search_cache(entry_id)
        if text_results is None:
            raise ValueError(f"No cache found for entry {entry_id}")

        # Apply fusion and similarity function
        return self._rerank_and_return_results(
            text_results, text_embed, img_embed, topk, fusion_similarity_fn
        )

    # @timeit
    def _rerank_and_return_results(
        self, text_results, text_embed, img_embed, topk, fusion_similarity_fn
    ):
        """
        Perform reranking using cached text_results and embeddings, first by text similarity
        (np.mean(np.max(text_sims, axis=1))) and then by fusion and similarity functions.
        """
        text_embed = normalize(text_embed)
        img_embed = normalize(img_embed)

        top_results = self.rank_based_on_text(text_embed, text_results)

        # Sort and get top 20 based on text similarity
        top_results.sort(key=lambda x: x[0], reverse=True)
        top_20_results = top_results[:20]
        top_20_results.sort(key=lambda x: x[0])

        # Step 2: Apply fusion similarity to the top 20 results
        final_results = []
        for res in top_20_results:
            filepath = res[1]
            text_sims = res[2]
            db_embed = res[3]
            if USE_GPU:
                # Convert embeddings to float32 tensors on GPU for image similarity
                if not isinstance(img_embed, torch.Tensor):
                    t_img = torch.tensor(img_embed, dtype=torch.float32, device="cuda")
                else:
                    t_img = img_embed.to("cuda").float()
                if not isinstance(db_embed, torch.Tensor):
                    t_db = torch.tensor(db_embed, dtype=torch.float32, device="cuda")
                else:
                    t_db = db_embed.to("cuda").float()
                img_sims = torch.matmul(t_img, t_db.T)
            else:
                img_sims = img_embed @ db_embed.T
            similarity = fusion_similarity_fn(text_sims, img_sims)
            # ensure native float for sorting
            if USE_GPU and isinstance(similarity, torch.Tensor):
                similarity = similarity.item()
            final_results.append((similarity, filepath))

        # Sort and return top_k based on fusion similarity
        final_results.sort(key=lambda x: x[0], reverse=True)
        final_top_results = final_results[:topk]
        top_filepaths = [r[1] for r in final_top_results]
        full_filepaths = [res[1] for res in final_results]

        return top_filepaths, full_filepaths

    # @timeit
    def rank_based_on_text(self, text_embed, text_results):
        # Step 1: Rank based on text similarity
        top_results = []
        for res in text_results:
            db_embed = res[3]
            filepath = res[2]
            db_embed = normalize(db_embed)
            if USE_GPU:
                # Convert embeddings to float32 tensors on GPU
                if not isinstance(text_embed, torch.Tensor):
                    t_text = torch.tensor(
                        text_embed, dtype=torch.float32, device="cuda"
                    )
                else:
                    t_text = text_embed.to("cuda").float()
                if not isinstance(db_embed, torch.Tensor):
                    t_db = torch.tensor(db_embed, dtype=torch.float32, device="cuda")
                else:
                    t_db = db_embed.to("cuda").float()
                text_sims = torch.matmul(t_text, t_db.T)
                text_rank_score = torch.mean(torch.max(text_sims, dim=1).values).item()
            else:
                text_sims = text_embed @ db_embed.T
                text_rank_score = np.mean(np.max(text_sims, axis=1))
            top_results.append((text_rank_score, filepath, text_sims, db_embed))
        return top_results


def load_dataset(json_path: str):
    with open(json_path, "r") as file:
        dataset = json.load(file)
    return dataset


def process_and_search(
    engine,
    dataset,
    topk=5,
    fusion_similarity_fn=None,
):
    """
    Process and search using only cached data to calculate recall.
    Evaluates on the test half of a 50/50 split (random_state=0) and
    returns Recall@topk.
    """
    score = 0
    top_100_score = 0
    mrr_score = 0
    processed = 0
    train, test = train_test_split(dataset, test_size=0.5, random_state=0)
    for i, entry in enumerate(test):
        entry_id = entry.get("No", i)  # Use 'No' field or index as entry_id
        question = entry["question"]
        options = entry["options"]

        prompt = question + "\n" + "\n".join(options)
        print(f"Searching for entry {entry_id}: {prompt}")

        # Perform the rerank with cached data and apply fusion and similarity functions
        top_results, full_filepaths = engine.rerank_from_cache(
            entry_id, topk=topk, fusion_similarity_fn=fusion_similarity_fn
        )

        print(f"Top search results for question {entry_id}: {top_results}")

        if exist(entry["pdf_path"], entry["pdf_image_page"], top_results):
            score += 1
        if exist(entry["pdf_path"], entry["pdf_image_page"], full_filepaths):
            top_100_score += 1
        mrr_score += calculate_mean_reciprocal_rank(entry, full_filepaths)
        processed += 1
        print(
            "Recall:",
            score / processed,
            "Full Recall:",
            top_100_score / processed,
            "MRR:",
            mrr_score / processed,
        )
    # After processing all test entries, return final recall score
    return score / processed


# "Text"   : CoPaLi (Text) baseline
# "Image"  : CoPaLi (Image) baseline
# "Fusion" : Patho-Fusion, Eq. (1) in the paper
similarity_metric = "Fusion"
print("Using similarity metric:", similarity_metric)


def fusion_similarity_fn(text_sim, img_sim) -> float:
    """
    Function to fuse pre-computed similarity matrices from different modalities.

    Arguments:
    - text_sim: np.ndarray of shape (N_t, N_d), S_t in the paper; N_t = number of query
                text tokens, N_d = number of document (page) tokens
    - img_sim : np.ndarray of shape (N_v, N_d), S_v in the paper; N_v = number of query
                image patches, N_d = number of document (page) tokens

    Returns:
    - similarity: Float, the fused similarity score for retrieval.
    """

    if similarity_metric == "Text":
        # Test Recall@1: 0.64 Recall@20: 0.92 MRR@20: 0.736
        return np.mean(np.max(text_sim, axis=1))
    elif similarity_metric == "Expert":
        # Test Recall@1: 0.62
        return np.mean(np.max(text_sim, axis=1)) - np.mean(np.mean(img_sim, axis=1))
    elif similarity_metric == "Image":
        return np.mean(np.max(img_sim, axis=1))
    else:
        assert similarity_metric == "Fusion", "Invalid similarity metric specified."
        # Eq. (1):
        #   std_i(std_j S_t[i, j]) * (mean_i kurt_j S_t[i, j])^2 * (mean_i kurt_j S_v[i, j])
        #   + mean_i max_j S_t[i, j]
        # Test Recall@1: 0.72 Recall@20: 0.92 MRR@20: 0.784
        return np.std(np.std(text_sim, axis=1)) * (
            np.mean(kurtosis(text_sim, axis=1)) ** 2
        ) * np.mean(kurtosis(img_sim, axis=1)) + np.mean(np.max(text_sim, axis=1))


if __name__ == "__main__":
    dataset_path = "rag_dataset/dataset.json"  # Path to your dataset JSON file
    engine = MilvusSearchEngine(
        cache_dir="cache"
    )  # Initialize with the cache directory

    # Load the dataset
    dataset = load_dataset(dataset_path)

    # Process and calculate recall using cached data, fusion, and similarity function
    print(
        "=== Processing and exploring recall using cached results with fusion and similarity function ==="
    )
    process_and_search(
        engine,
        dataset,
        topk=top_k,
        fusion_similarity_fn=fusion_similarity_fn,
    )
