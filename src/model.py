import numpy as np


def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def sample_negative_words(
    probs: np.ndarray,
    k: int,
    positive_id: int
) -> np.ndarray:
    size = max(2 * k, 8)

    while True:
        neg_ids = np.random.choice(len(probs), size=size, p=probs)
        neg_ids = neg_ids[neg_ids != positive_id]

        if len(neg_ids) >= k:
            return neg_ids[:k].astype(np.int32)

        size *= 2


def training_step(
    center_id: int,
    context_id: int,
    W_in: np.ndarray,
    W_out: np.ndarray,
    lr: float,
    num_negative: int,
    neg_probs: np.ndarray
) -> float:
    v_c = W_in[center_id].copy()
    u_o = W_out[context_id].copy()

    neg_ids = sample_negative_words(neg_probs, num_negative, context_id)
    u_neg = W_out[neg_ids].copy()

    # Forward pass
    pos_score = np.dot(v_c, u_o)
    neg_scores = np.dot(u_neg, v_c)

    pos_sig = sigmoid(pos_score)
    neg_sig = sigmoid(neg_scores)

    # Loss
    loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(sigmoid(-neg_scores) + 1e-10))

    # Gradients
    grad_v = (pos_sig - 1.0) * u_o + np.sum(neg_sig[:, None] * u_neg, axis=0)
    grad_uo = (pos_sig - 1.0) * v_c
    grad_uneg = neg_sig[:, None] * v_c[None, :]

    # Parameter updates
    W_in[center_id] -= lr * grad_v
    W_out[context_id] -= lr * grad_uo
    W_out[neg_ids] -= lr * grad_uneg

    return float(loss)