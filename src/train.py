import numpy as np

from src.model import training_step


def train_word2vec(
    corpus_ids: np.ndarray,
    vocab_size: int,
    neg_probs: np.ndarray,
    embedding_dim: int = 50,
    window_size: int = 2,
    epochs: int = 2,
    lr: float = 0.025,
    num_negative: int = 5
):
    W_in = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01
    W_out = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01

    n = len(corpus_ids)
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for i in range(n):
            center_id = corpus_ids[i]

            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)

            for j in range(left, right):
                if j == i:
                    continue

                context_id = corpus_ids[j]

                loss = training_step(
                    center_id=center_id,
                    context_id=context_id,
                    W_in=W_in,
                    W_out=W_out,
                    lr=lr,
                    num_negative=num_negative,
                    neg_probs=neg_probs,
                )

                total_loss += loss
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, avg loss = {avg_loss:.4f}")

    return W_in, W_out, losses