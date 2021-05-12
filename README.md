# Insight
👁️👄👁️ Natural language document search. Given a topic query, find the `n` most similar documents.  
A `DistilBERT` or `SciBERT` model is used to embed the query and the documents.

## 🏡 Getting Started
To run, first install the requirements in your virtual environment:

`pip install -r requirements.txt`

Then run `search.py`:

`python3 search.py -q "machine learning for enzyme interaction predictions"`

With flags:
```
MANDATORY:
-----------
-q        |   (str) The topic / query string

OPTIONAL:
-----------
-n        |   (int) The number of most similar results to return
-scibert  |   (bool) Whether to use SciBERT, not DistilBERT (NOTE: Very compute heavy! May kill your process.)
-train    |   (bool) Whether to finetune model first [NOT IMPLEMENTED]
-docpath  |   (str) The filepath of the docs to finetune on [NOT IMPLEMENTED
```

## 🔗 Links

`DistilBERT` model taken from [this 🤗 Hugging Face repo](https://huggingface.co/distilbert-base-uncased).
`SciBERT` model taken from [this 🤗 Hugging Face repo](https://huggingface.co/allenai/scibert_scivocab_uncased).

## ✔️ TODO

- [x] Allow SciBERT instead of DistilBERT optionality.
- [ ] Do a single-round of fine-tuning the model on UKRI grant data.
- [ ] Return the document title, not the most relevant sentence.
- [ ] Store latent embeddings of documents separately so you don't have to recompute each query.
- [ ] Incorporate active learning classification step. (Important)

