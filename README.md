# Word2Vec from Scratch (NumPy)

This is a from-scratch implementation of the Skip-gram Word2Vec model using only NumPy.

The goal was to go through the full pipeline manually from raw text to trained embeddings and understand how Word2Vec actually works, without relying on libraries like PyTorch or TensorFlow.

---

## What’s implemented

- Skip-gram model  
- Negative sampling (`p(w) ∝ f(w)^0.75`)  
- Subsampling of frequent words  
- Manual gradient computation and parameter updates  
- Cosine similarity for inspecting learned embeddings  

The loss function and negative sampling formulation are taken from the original Word2Vec paper (Mikolov et al.) and implemented directly.

---

## Dataset

The model is trained on the **text8 dataset** (a cleaned Wikipedia corpus):

- lowercase text  
- no punctuation  
- space-separated tokens  

Training is done on the first **1 million tokens**.

After preprocessing:
- ~506k tokens after subsampling  
- ~444k tokens after filtering  
- vocabulary size ≈ 14k  

---

## Training

- embedding dimension: 50  
- window size: 2  
- negative samples: 5  
- epochs: 2  
- learning rate: 0.025
 
I used embedding dimension 50 and window size 2 to keep training efficient while still capturing local context.
Larger dimensions or windows would improve semantic quality, but at a higher computational cost. 

Training is done with a simple SGD loop (no batching).
