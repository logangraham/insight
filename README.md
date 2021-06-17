# Insight
👁️👄👁️ Natural language document search. Given a topic query, find the `n` most similar documents.  
A `DistilBERT` or `SciBERT` model is used to embed the query and the documents.

## 🏡 Getting Started
To run, first install the requirements in your virtual environment:

`pip install -r requirements.txt`

Then run `streamlit run app.py`, type in your query, and hit cmd/ctrl+enter.

Alternatively, you can use the manifest and Procfile to push to your PaaS platform.

You'll need the metadata (`metadata.json`) and embedding (`doctensor.pt`) files. Ask me :)

## 🔗 Links

`DistilBERT` model taken from [this 🤗 Hugging Face repo](https://huggingface.co/distilbert-base-uncased).

`SciBERT` model taken from [this 🤗 Hugging Face repo](https://huggingface.co/allenai/scibert_scivocab_uncased).

## ✔️ TODO

Improve retrieval performance
- [ ] Try using a different embedding layer (2nd to last?)
- [ ] Try with [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [ ] Try out full BERT, Doc2Vec, [GPT-neo](https://huggingface.co/EleutherAI/gpt-neo-125M), Word2Vec, and ensemble
- [ ] Experiment with mixture of experts model, e.g. bio & CS papers handled by SciBERT or BioBERT.
- [ ] Sort out why BERT models prefers small/short abstracts. (padding?)
- [ ] Experiment: find an optimal distance metric.
- [ ] Experiment: break abstracts into 75-word chunks. Take maximally related chunks.
- [x] [Fine-tune on grant data.](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py) [2](https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379)
- [x] Allow SciBERT instead of DistilBERT optionality.
- [x] Experiment: use full abstract embeddings instead of sentence embeddings.

Classify into categories
- [ ] Add active learning classification step. (Important)

Other
- [ ] Add year filters.
- [x] Deploy to prod.
- [x] Add minimum word count (~100 = 85% of abstracts).
- [x] Add spark lines.