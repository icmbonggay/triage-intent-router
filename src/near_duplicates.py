"""Group messages that are duplicates or near-duplicates of each other.

How it works, in three steps:

1. Represent: each normalized message becomes a TF-IDF vector built from
   character 3- to 5-grams, so small typos ("arived" vs "arrived") still
   share most of their pieces.
2. Search: NearestNeighbors with the cosine metric finds, for every
   message, the other messages whose similarity is at least `threshold`.
   It never builds the full n x n table, so it scales past Banking77.
3. Group: every similar pair is an edge in a graph. Each connected
   component of that graph is one group, so if A~B and B~C, then A, B and
   C share a group even when A and C are not similar on their own.
"""

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from duplicates import assign_groups
from normalize import normalize_data

# PROVISIONAL: confirm by running scripts/choose_threshold.py on the real
# data, reading the pairs, and updating this value with the reason in the README.
DEFAULT_THRESHOLD = 0.9


def find_similar_pairs(
    texts: list[str], threshold: float
) -> list[tuple[int, int, float]]:
    """Find every pair of texts whose cosine similarity is at least `threshold`.

    Texts are normalized first. Texts that normalize to an empty string
    (for example "???") have no characters to compare, so they are skipped
    here; exact matching in `assign_near_duplicate_groups` still groups them.

    Args:
        texts: The messages to compare.
        threshold: Minimum cosine similarity, between 0 and 1, for a pair
            to count as near-duplicates.

    Returns:
        A list of (i, j, similarity) tuples with i < j, where i and j are
        positions in `texts`. Each pair appears once.
    """
    normalized = [normalize_data(text) for text in texts]
    positions = [i for i, text in enumerate(normalized) if text]
    if len(positions) < 2:
        return []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vectors = vectorizer.fit_transform([normalized[i] for i in positions])

    # Cosine distance is 1 - cosine similarity, so "similarity >= threshold"
    # is the same as "distance <= 1 - threshold".
    search = NearestNeighbors(metric="cosine", radius=1 - threshold)
    search.fit(vectors)
    distances, neighbors = search.radius_neighbors(vectors)

    pairs = []
    for a, (row_distances, row_neighbors) in enumerate(zip(distances, neighbors)):
        for distance, b in zip(row_distances, row_neighbors):
            if a < b:  # skip self-matches and keep each pair once
                pairs.append((positions[a], positions[b], 1 - distance))
    return pairs


def groups_from_pairs(n: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Turn similar pairs into groups using connected components.

    Args:
        n: The total number of items, so items with no pair still get a group.
        pairs: (i, j) position pairs that should end up in the same group.

    Returns:
        A list of n integer group labels. Two items share a label exactly
        when a chain of pairs connects them.
    """
    if n == 0:
        return []

    rows = [i for i, _ in pairs]
    cols = [j for _, j in pairs]
    graph = coo_matrix(([1] * len(pairs), (rows, cols)), shape=(n, n))

    _, labels = connected_components(graph, directed=False)
    return labels.tolist()


def assign_near_duplicate_groups(
    df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> list[int]:
    """Label each row so that duplicates and near-duplicates share a label.

    Exact duplicates (identical after `normalize_data`) are always grouped.
    Near-duplicates are grouped when their similarity is at least
    `threshold`, and chains of near-duplicates end up in one group.

    Args:
        df: DataFrame with a "text" column holding the messages.
        threshold: Minimum cosine similarity for two messages to count as
            near-duplicates.

    Returns:
        A list with one integer group label per row of `df`, in row order.
    """
    texts = df["text"].tolist()

    # Edges for exact duplicates: link every row to the first row with
    # the same normalized text. That first row is the group's representative.
    first_position = {}
    exact_pairs = []
    for position, label in enumerate(assign_groups(df)):
        if label in first_position:
            exact_pairs.append((first_position[label], position))
        else:
            first_position[label] = position

    # Compare only one representative per exact group. Extra copies would
    # count the same words several times when TF-IDF computes its weights,
    # which skews the similarity scores, and they add no new information.
    representatives = list(first_position.values())
    representative_texts = [texts[p] for p in representatives]
    near_pairs = [
        (representatives[i], representatives[j])
        for i, j, _ in find_similar_pairs(representative_texts, threshold)
    ]

    return groups_from_pairs(len(texts), exact_pairs + near_pairs)
