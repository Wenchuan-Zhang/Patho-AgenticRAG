import re
import math


def extract_page(filepath):
    """
    Extracts the page number from a filepath of the form "..._page_<number>.jpg".
    Returns the page number as an integer, or None if not found.
    """
    match = re.search(r"_page_(\d+)", filepath)
    return int(match.group(1)) if match else None


def relevance_score(page, ideal_page):
    """
    Assigns a relevance score based on the distance from the ideal page:
      - 2 if exactly the ideal page (very relevant)
      - 1 if the page is ±1 from the ideal page (somewhat relevant)
      - 0 otherwise (irrelevant)
    """
    if page == ideal_page:
        return 2
    elif abs(page - ideal_page) == 1:
        return 1
    else:
        return 0


def ndcg_at_k(full_filepaths, ideal_page, k=None):
    """
    Calculates nDCG@k for a list of image filepaths given an ideal page number.

    Parameters:
    - full_filepaths: List[str], filepaths containing "_page_<num>"
    - ideal_page: int, the ground-truth page number
    - k: int or None, the cutoff for ranking positions (defaults to all)

    Returns:
    - ndcg: float, the normalized Discounted Cumulative Gain at k
    """
    # Extract relevance scores in ranking order
    relevances = []
    for filepath in full_filepaths:
        page = extract_page(filepath)
        score = relevance_score(page, ideal_page) if page is not None else 0
        relevances.append(score)

    # Limit to top-k
    if k is None or k > len(relevances):
        k = len(relevances)
    relevances = relevances[:k]

    # Compute DCG
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))

    # Compute IDCG (ideal DCG)
    ideal_rels = sorted(relevances, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0


