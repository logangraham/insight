import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM
import streamlit as st
import torch
from search import *
from helpers import *


st.set_page_config(page_title="UK Science R&D Spending Search")

## Expensive Functions
@st.cache
def load_embeddings(embeddings_path="./data/distilbert3tensor.pt"):
    M = torch.load(embeddings_path)
    return M

@st.cache(allow_output_mutation=True)
def load_model(path_or_name='./model/distilbert3/'):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForMaskedLM.from_pretrained(path_or_name)
    return tokenizer, model

@st.cache
def load_indices():
    idx = torch.load("./data/chunkindices.pt")
    return idx

def main():
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)

    st.title("What 🔬 science do we fund?")
    st.write('This search engine helps you answer the question: _"How much are we funding X?"_, where X is some topic. The topic is anything you can describe in words. The engine uses deep learning models to find grants that are closest to your topic.')

    # define query
    query = st.text_area("Topic (hint: use at least 5-10 words to describe your topic)", "")
    embeddings = load_embeddings()
    # use_sentences = st.checkbox("Use multi-sentence embeddings? (Instead of document embeddings)")

    # load preamble
    # if use_sentences:
    #     embeddings = load_embeddings("./data/chunktensor.pt")
    #     idx = load_indices()
    # else:
    #     embeddings = load_embeddings()
    tokenizer, model = load_model()

    # fetch results
    if query:
        # parameters
        col1, col2, col3 = st.beta_columns(3)
        # rank by similarity or size
        with col1:
            rank_strategy = st.selectbox("Sort by", ["Similarity", "£ value"])
        # select number of relevant papers
        with col2:
            num_results = st.slider("Number of results", 25, 1000, value=50, step=25)
        with col3:
            min_words = st.slider("Min. words in abstract", 1, 250, value=100, step=1)

        # fetch results
        # TODO: Fix sentence segment search
        # if use_sentences:
        #     results = return_ranked_by_sentence(query, tokenizer, model, idx, embeddings)
        # else:
        #     results = return_ranked(query, tokenizer, model, embeddings)
        results = return_ranked(query, tokenizer, model, embeddings)

        # subset to enough words
        results = [r for r in results if len(metadata[str(r[0])]['abstract'].split()) > min_words]

        # return data
        meta = [{"title": metadata[str(i)]["project_title"],
                "value": int(metadata[str(i)]["value"]),
                "n_words": len(metadata[str(i)]["abstract"].split()),
                "date": int(datetime.strptime(metadata[str(i)]['start_date'],
                                              "%d/%m/%Y %H:%M").year),
                "distance": dist}
                for i, dist in results[:num_results]]

        # sort for printing
        if rank_strategy == "£ value":
            meta = sorted(meta, key=lambda x: -x["value"])

        # get total
        total = sum([el["value"] for el in meta])


        #get sparklines
        _1, spark, _3 = st.beta_columns(3)
        with spark:

            st.markdown(f"""
            \
            \
            💵 Total spent
            # £{total:,}
            """)

            spark_data = [sum([el["value"] for el in meta if el["date"] == year]) for year in range(2015, 2022)]
            st.pyplot(sparkline(spark_data))


        st.write(f"""
        # Top Grants
        """)
        
        write_paper_table(meta)


if __name__ == "__main__":
    main()