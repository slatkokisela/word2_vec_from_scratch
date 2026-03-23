
# Word2Vec from scratch (NumPy)

This is an implementation of the Skip-gram Word2Vec model using only NumPy.

The goal was to go through the full pipeline manually from raw text to trained embeddings and understand how Word2Vec actually works, without relying on libraries like PyTorch or TensorFlow.

---

## What’s implemented

- Skip-gram model  
- Negative sampling (`p(w) ∝ f(w)^0.75`)  
- Subsampling of frequent words  
- Manual gradient computation and parameter updates  
- Cosine similarity for inspecting learned embeddings  

The loss function and negative sampling formulation are taken from the original Word2Vec paper (Mikolov et al.).

---

## Dataset

The model is trained on the **text8 dataset** (a cleaned Wikipedia corpus):

- lowercase text  
- no punctuation  
- space separated tokens  

Training is done on the first **1 million tokens**.

After preprocessing:
- ~506k tokens after subsampling  
- ~444k tokens after filtering  
- vocabulary size ≈ 14k  

To run the project, download the dataset from:
http://mattmahoney.net/dc/text8.zip

Unzip it and place the file in:

data/text8


The dataset is not included in this repository due to its size.

---

## Training

- embedding dimension: 100  
- window size: 3  
- negative samples: 5  
- epochs: 2  
- learning rate: 0.025  

I kept the embedding size relatively small and the window narrow to keep training fast, while still capturing local context. 
Larger values would improve the quality of embeddings, but would also make training significantly slower.

Training is done with a simple SGD loop (no batching in simple version).

---

## Experiments

I tried a couple of simple changes to see how they affect the results.

First I increased the embedding dimension from 50 to 100 and the context window from 2 to 3. 
This led to lower training loss and slightly more meaningful nearest neighbors for some words (examples "king" and "city")

Then I increased the number of epochs from 2 to 3:

Epoch 1 avg loss: 2.8490  
Epoch 2 avg loss: 2.4118  
Epoch 3 avg loss: 2.3168  

The loss kept decreasing, but the quality of the results didn’t really improve at all. In some cases, the nearest neighbors became more noisy and more under hallucination.

So even though the model continues to optimize the objective, it doesn’t necessarily mean the embeddings are getting better. 
With this setup dataset and computing power, 2 epochs were the best choice.

---

## Notes

Implementing everything from scratch made it much clearer how negative sampling, subsampling, and gradient updates interact in practice, beyond just the highlevel description from the paper.
