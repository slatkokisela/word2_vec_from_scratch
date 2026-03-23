import numpy as np

from src.data import (
    read_text,
    subsample_tokens,
    build_vocab,
    tokens_to_ids,
    build_negative_sampling_distribution,
    save_outputs,
)
from src.train import train_word2vec
from src.utils import most_similar


def main():
    np.random.seed(42)

    data_path = "data/text8"
    output_dir = "outputs"

    max_tokens = 1_000_000
    min_count = 5
    subsample_t = 1e-4
    embedding_dim = 100
    window_size = 3
    epochs = 2
    learning_rate = 0.025
    num_negative = 5

    text = read_text(data_path)
    tokens = text.split()[:max_tokens]

    print("Original token count:", len(tokens))

    tokens = subsample_tokens(tokens, t=subsample_t)
    print("Token count after subsampling:", len(tokens))

    filtered_tokens, word_to_id, id_to_word, counts = build_vocab(tokens, min_count=min_count)
    corpus_ids = tokens_to_ids(filtered_tokens, word_to_id)
    neg_probs = build_negative_sampling_distribution(counts, word_to_id)

    print("Vocabulary size:", len(word_to_id))
    print("Corpus length after filtering:", len(corpus_ids))

    W_in, W_out, losses = train_word2vec(
        corpus_ids=corpus_ids,
        vocab_size=len(word_to_id),
        neg_probs=neg_probs,
        embedding_dim=embedding_dim,
        window_size=window_size,
        epochs=epochs,
        lr=learning_rate,
        num_negative=num_negative,
    )

    save_outputs(
        output_dir=output_dir,
        W_in=W_in,
        W_out=W_out,
        losses=losses,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
    )

    print("\nLosses by epoch:", losses)

    for word in ["king", "queen", "city", "man", "woman"]:
        most_similar(word, W_in, word_to_id, id_to_word, top_k=5)


if __name__ == "__main__":
    main()
