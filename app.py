import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
import streamlit as st
import torch


st.set_page_config(page_title="UK Science R&D Spending Search Engine")

## Expensive Functions
@st.cache
def load_embeddings():
    M = torch.load("data/doctensor.pt")
    return M

@st.cache
def load_indices():
    idx = torch.load("data/indices.pt")
    return idx

@st.cache(allow_output_mutation=True)
def load_model(model_name='distilbert'):
    if model_name == "scibert":
        model_id = "allenai/scibert_scivocab_uncased"
    else:
        model_id = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    return tokenizer, model

# don't cache because query will change every time
def embed(query, tokenizer, model):
    token = tokenizer([query], return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**token, output_hidden_states=True).hidden_states[-1]
    ## use pooling across vocab size
    query_embedding = mean_pooling(query_embedding, token['attention_mask'])
    return query_embedding


## Not expensive functions
def mean_pooling(token_embeddings, attention_mask):
    """
    Effectively averages the embeddings of tokens across the vocabulary dimension
    to calculate the vocab-weighted latent representations (embeddings).

    :param token_embeddings: torch.float tensor of size (n_examples, n_vocab, n_latent)
    :param attention_mask: torch.byte tensor of size (n_examples, n_vocab)
    :return: torch.float tensor of size (n_examples, n_latent)
    """
    return torch.mean(token_embeddings, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def cosine_similarity(v, M):
    """
    L2 similarity between a vector (single query embedding) and a matrix (of embeddings).

    :param v: torch.tensor of size (1, n_latent)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: torch.tensor of size (n_documents,)
    """
    dv = v.norm(p=2)
    dM = M.norm(p=2, dim=1)
    return (M.matmul(v.T)).squeeze().div(dM * dv)

def L2_similarity(v, M):
    return -torch.cdist(M, v).squeeze()

def search(query, tokenizer, model, M):
    q = embed(query, tokenizer, model)
    sims = L2_similarity(q, M)
    rankings = torch.argsort(sims, descending=True)
    result = rankings.tolist()
    return result

## Write app
def write_paper_table(data, n_words=True):
    table_md = f"""
    |Rank|Title|Value|{"# words|"*n_words}
    |--|--|--|--|
    """
    for i, el in enumerate(data):
        table_md += f"""|{i+1}|**{el[0]}**|£{el[1]:,}|{(str(el[2]) + "|")*n_words}
        """
    st.markdown(table_md)

def main():
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)

    st.title("What 🔬 science do we fund?")
    st.write('This search engine helps you answer the question: _"How much are we funding X?"_, where X is some topic. The topic is anything you can describe in words. The engine uses sophisticated machine learning models to find grants that are closest to your topic.')

    # load preamble
    embeddings = load_embeddings()
    tokenizer, model = load_model()

    # define query
    query = st.text_area("Topic", "")

    # fetch results
    if query:
        # fetch results
        results = search(query, tokenizer, model, embeddings)
        
        # parameters
        col1, col2, col3 = st.beta_columns(3)
        # rank by similarity or size
        with col1:
            rank = st.selectbox("Sort by", ["Similarity", "£ value"])
        # select number of relevant papers
        with col2:
            num_results = st.slider("Number of results", 10, 100, value=5, step=1)
        with col3:
            min_words = st.slider("Min. words in abstract", 100, 250, value=100, step=1)

        # return data
        meta = [(metadata[str(i)]["project_title"],
                int(metadata[str(i)]["value"]),
                int(len(metadata[str(i)]["abstract"].split())))
                for i in results if len(metadata[str(i)]['abstract'].split()) > min_words]
        meta = meta[:num_results]

        total = sum([el[1] for el in meta])

        if rank == "£ value":
            meta = sorted(meta, key=lambda x: -x[1])

        st.markdown(f"""
        \
        \
        # £{total:,}
        💵 Total spent
        """)


        st.write(f"""
        # Top Grants
        """)
        
        write_paper_table(meta)


if __name__ == "__main__":
    main()