import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def most_similar(
    word: str,
    W_in: np.ndarray,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    top_k: int = 5
):
    if word not in word_to_id:
        print(f"Word '{word}' not in vocabulary.")
        return

    query_id = word_to_id[word]
    query_vec = W_in[query_id]

    sims = []
    for idx in range(W_in.shape[0]):
        if idx == query_id:
            continue
        sim = cosine_similarity(query_vec, W_in[idx])
        sims.append((id_to_word[idx], sim))

    sims.sort(key=lambda x: x[1], reverse=True)

    print(f"\nMost similar to '{word}':")
    for neighbor, score in sims[:top_k]:
        print(f"  {neighbor:20s} {score:.4f}")