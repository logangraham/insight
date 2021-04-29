# Insight
Natural language document search. Given a topic query, find the `n`  
most similar documents. A `DistilBERT` model is used to embed the query
and the documents.

## Getting Started
To run, first install the requirements in your virtual environment:

`pip install requirements.txt`

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
-scibert  |   (bool) Whether to use SciBERT, not DistilBERT [NOT IMPLEMENTED]
-train    |   (bool) Whether to finetune model first [NOT IMPLEMENTED]
-docpath |   (str) The filepath of the docs to finetune on [NOT IMPLEMENTED
```

## TODO:

[ ] Use SciBERT instead of DistilBERT optionality.
[ ] Do a single-round of fine-tuning the model on UKRI grant data.
[ ] Return the document title, not the most relevant sentence.
[ ] Store latent embeddings of documents separately so you don't have to recompute each query.

