import json

import numpy as np
import torch
from llm4ad.base import Evaluation
from sklearn.model_selection import train_test_split

from multimodal_fusion.ndcg import ndcg_at_k
from multimodal_fusion.time_util import timeit
from search_engine.offline_rerank import MilvusSearchEngine, exist, top_k
from multimodal_fusion.basic_functions import USE_GPU

template_program = '''
import numpy as np
def fusion_similarity_fn(text_embed, img_embed, db_embed)->float:
    """
    Combined function for fusion and similarity calculation.

    Arguments:
    - text_embed: np.ndarray of shape (T, D), where T = number of text tokens
    - img_embed : np.ndarray of shape (I, D), where I = number of image patches/features
    - db_embed  : np.ndarray of shape (N, D), where N = number of tokens for a database item

    Returns:
    - similarity: Float, computed similarity between the fused embeddings and db_embed
    """
    text_sims = text_embed @ db_embed.T
    img_sims = img_embed @ db_embed.T

    # Compute cosine similarity between fused embedding and db_embed
    text_sims = text_embed @ db_embed.T
    img_sims = img_embed @ db_embed.T
    sim = np.mean(text_sims.max(1)) - np.mean(img_sims.mean(1))

    return max_sim
'''

task_description = """
In this task, you are provided with three sets of embeddings: text, image, and database embeddings. Your goal is to develop a robust retrieval system that leverages both textual and visual information by calculating similarities for each modality separately and then fusing these similarities to improve retrieval accuracy.

Specifically, you should compute the similarity between the text embeddings and the database embeddings, as well as between the image embeddings and the database embeddings. After obtaining these similarity scores, you are required to design a novel similarity fusion algorithm that intelligently combines the similarity values from each modality to produce a final, unified similarity score for retrieval.

The challenge lies in evolving the fusion function beyond the baseline approach of
`sim = np.mean(text_sims.max(1)) - np.mean(img_sims.mean(1))`.
Your fusion algorithm should more effectively capture the complementary relevance of each modality to improve the ranking of database entries, thereby enhancing retrieval performance.

You are to implement a function that takes text, image, and database embeddings as input, calculates similarity scores for each modality, and fuses these scores using your designed algorithm to output a final similarity value. Retrieval should be performed by ranking the database entries according to these fused similarity scores.
"""

template_program = '''
import numpy as np
def fusion_similarity_fn(text_sim, img_sim)->float:
    """
    Function to fuse pre-computed similarity matrices from different modalities.

    Arguments:
    - text_sim: np.ndarray of shape (T, N), where T = number of text tokens, N = number of database items
    - img_sim : np.ndarray of shape (I, N), where I = number of image patches/features, N = number of database items

    Returns:
    - similarity: Float, the fused similarity score for retrieval.
    """
    sim = np.mean(np.max(text_sims,axis=1))

    return sim
'''

task_description = """
In this task, you are provided with pre-computed similarity matrices for text and image modalities. Your objective is to design a **similarity merge function** that fuses the given similarities into a single score for retrieval. You do not need to compute the similarity matrices themselves—your sole focus is to combine them effectively.

You must implement the merge function using no more than 5 lines of code. The baseline is:

```
sim = np.mean(np.max(text_sim, axis=1)) - np.mean(np.mean(img_sim, axis=1))
```

-np.mean(np.mean(img_sim, axis=1)) to encourage the only a specific region has large similarity, to filter out very general image 

Your merge function should improve upon this baseline by more effectively capturing the relevance of each modality. The final function should take `text_sim` and `img_sim` as input and output a single, fused similarity score. Retrieval is based on ranking database entries by this score.
"""


class SimilarityEvaluation(Evaluation):
    """Evaluator for similarity-based search."""

    def __init__(
        self,
        dataset_path="rag_dataset/dataset.json",
        topk=top_k,
        use_cache=True,
        fusion_similarity_fn=None,
        metric="ndcg",  # default changed here
        **kwargs,
    ):
        """
        Args:
            dataset_path (str): Path to the dataset JSON file.
            engine (MilvusSearchEngine): Search engine instance.
            topk (int): Number of top results to consider for evaluation.
            use_cache (bool): Whether to use cache.
            fusion_similarity_fn (callable): Similarity function to calculate score between embeddings.
        """
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=120,
            **kwargs,
        )
        self.dataset_path = dataset_path
        self.engine = MilvusSearchEngine()
        self.topk = topk
        self.use_cache = use_cache
        self.fusion_similarity_fn = fusion_similarity_fn
        self.dataset = self.load_dataset()
        self.metric = metric.lower()  # normalize for comparisons
        print("Metric for evaluation:", self.metric)

    def load_dataset(self):
        """Load the dataset from a JSON file."""
        with open(self.dataset_path, "r") as file:
            dataset = json.load(file)
        return dataset

    @timeit
    def evaluate_program(
        self, program_str, callable_func, return_semantics=False, indices=None
    ):
        # If using GPU, wrap the fusion function to return numpy
        if USE_GPU:

            def _fusion_fn(text_embed, img_embed, *args, **kwargs):
                out = callable_func(text_embed, img_embed, *args, **kwargs)
                if isinstance(out, torch.Tensor):
                    return out.detach().cpu().numpy()
                return out

            fusion_fn = _fusion_fn
        else:
            fusion_fn = callable_func
        recall_scores, mrr_scores, ndcg_scores, full_recall_scores = [], [], [], []
        processed = 0

        # Iterate over a subset of the dataset to evaluate
        train, test = train_test_split(self.dataset, test_size=0.5, random_state=0)
        if indices is not None:
            train = [train[i] for i in indices]
            print("Number of entries in train set:", len(train))
        for i, entry in enumerate(train):
            entry_id = entry.get("No", i)
            # page_range = tuple(int(x) for x in entry["pdf_page"].split("-"))

            # Perform rerank from cache using the search engine
            top_results, full_filepaths = self.engine.rerank_from_cache(
                entry_id, topk=self.topk, fusion_similarity_fn=fusion_fn
            )

            # compute each metric
            recall_scores.append(self.calculate_recall(entry, top_results))
            mrr_scores.append(self.calculate_mean_reciprocal_rank(entry, top_results))
            full_recall_scores.append(self.calculate_recall(entry, full_filepaths))

            # NDCG based on page ±1 relevance
            ndcg_scores.append(
                ndcg_at_k(
                    full_filepaths, ideal_page=int(entry["pdf_image_page"]), k=self.topk
                )
            )
            processed += 1

        # Compute average recall
        avg_recall = np.mean(recall_scores)
        avg_mrr = np.mean(mrr_scores)
        avg_full_recall = np.mean(full_recall_scores)
        avg_ndcg = np.mean(ndcg_scores)

        print(
            f"Program: {program_str}, "
            f"Avg Recall: {avg_recall:.4f}, "
            f"Avg MRR: {avg_mrr:.4f}, "
            f"Avg Full Recall: {avg_full_recall:.4f}, "
            f"Avg NDCG: {avg_ndcg:.4f}"
        )
        if return_semantics:
            if self.metric == "mrr":
                return avg_mrr, mrr_scores
            elif self.metric == "recall":
                return avg_recall, recall_scores
            else:  # ndcg
                return avg_ndcg, ndcg_scores
        else:
            if self.metric == "mrr":
                return avg_mrr
            elif self.metric == "recall":
                return avg_recall
            else:  # ndcg
                return avg_ndcg

    def calculate_recall(self, entry, top_results):
        """Calculate recall for an entry using the eva similarity function."""
        if exist(entry["pdf_path"], entry["pdf_image_page"], top_results):
            return 1
        return 0

    def calculate_mean_reciprocal_rank(self, entry, top_results):
        for idx, item in enumerate(top_results):
            if exist(entry["pdf_path"], entry["pdf_image_page"], [item]):
                # If relevant item is found, return the reciprocal of the rank
                return 1 / (idx + 1)
        # If no relevant item is found, return 0
        return 0
