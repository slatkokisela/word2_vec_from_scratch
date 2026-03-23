import os
from collections import Counter

import numpy as np


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def subsample_tokens(tokens: list[str], t: float = 1e-5) -> list[str]:
    counts = Counter(tokens)
    total = len(tokens)
    freqs = {word: count / total for word, count in counts.items()}

    kept_tokens = []
    for word in tokens:
        f = freqs[word]
        p_discard = 1.0 - np.sqrt(t / f)
        p_discard = max(0.0, p_discard)

        if np.random.rand() > p_discard:
            kept_tokens.append(word)

    return kept_tokens


def build_vocab(tokens: list[str], min_count: int = 5):
    counts = Counter(tokens)

    filtered_words = [word for word, count in counts.items() if count >= min_count]
    filtered_words.sort()

    word_to_id = {word: idx for idx, word in enumerate(filtered_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    filtered_tokens = [word for word in tokens if word in word_to_id]

    return filtered_tokens, word_to_id, id_to_word, counts


def tokens_to_ids(tokens: list[str], word_to_id: dict[str, int]) -> np.ndarray:
    return np.array([word_to_id[word] for word in tokens], dtype=np.int32)


def build_negative_sampling_distribution(
    counts: Counter,
    word_to_id: dict[str, int]
) -> np.ndarray:
    vocab_size = len(word_to_id)
    freqs = np.zeros(vocab_size, dtype=np.float64)

    for word, idx in word_to_id.items():
        freqs[idx] = counts[word]

    probs = freqs ** 0.75
    probs_sum = probs.sum()
    if probs_sum == 0:
        raise ValueError("Negative sampling distribution has zero sum.")
    probs /= probs_sum

    return probs


def save_outputs(
    output_dir: str,
    W_in: np.ndarray,
    W_out: np.ndarray,
    losses: list[float],
    word_to_id: dict[str, int],
    id_to_word: dict[int, str]
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "W_in.npy"), W_in)
    np.save(os.path.join(output_dir, "W_out.npy"), W_out)

    import json

    with open(os.path.join(output_dir, "losses.json"), "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=2)

    with open(os.path.join(output_dir, "word_to_id.json"), "w", encoding="utf-8") as f:
        json.dump(word_to_id, f, indent=2, ensure_ascii=False)

    id_to_word_str = {str(k): v for k, v in id_to_word.items()}
    with open(os.path.join(output_dir, "id_to_word.json"), "w", encoding="utf-8") as f:
        json.dump(id_to_word_str, f, indent=2, ensure_ascii=False)