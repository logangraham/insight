# Insight
👁️👄👁️ Natural language document search. Given a topic query, find the `n` most similar documents.  
A `DistilBERT` or `SciBERT` model is used to embed the query and the documents.

## 🏡 Getting Started
To run, first install the requirements in your virtual environment:

`pip install -r requirements.txt`

Then run `streamlit run app.py`, type in your query, and hit cmd/ctrl+enter.

## 🔗 Links

`DistilBERT` model taken from [this 🤗 Hugging Face repo](https://huggingface.co/distilbert-base-uncased).
`SciBERT` model taken from [this 🤗 Hugging Face repo](https://huggingface.co/allenai/scibert_scivocab_uncased).

## ✔️ TODO

- [x] Allow SciBERT instead of DistilBERT optionality.
- [ ] Do a single-round of fine-tuning grant data.
- [x] Experiment: use full abstract embeddings instead of sentence embeddings.
- [ ] Experiment: find an optimal distance metric.
- [ ] Experiment: average sentence vectors into document vectors.
- [ ] Incorporate active learning classification step. (Important)