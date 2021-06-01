import torch
import numpy as np


def mean_pooling(token_embeddings, attention_mask):
    """
    Effectively averages the embeddings of tokens across the vocabulary dimension
    to calculate the vocab-weighted latent representations (embeddings).

    :param token_embeddings: torch.float tensor of size (n_examples, n_vocab, n_latent)
    :param attention_mask: torch.byte tensor of size (n_examples, n_vocab)
    :return: torch.float tensor of size (n_examples, n_latent)
    """
    # return torch.mean(token_embeddings, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed(query, tokenizer, model):
    """
    Embed `query` using `model` and return it.

    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    """
    token = tokenizer([query], return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**token, output_hidden_states=True).hidden_states[-1]
    ## use pooling across vocab size
    query_embedding = mean_pooling(query_embedding, token['attention_mask'])
    return query_embedding

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
    """
    Calculate the L2 / Euclidean distance between all rows in M and v.
    
    :param v: torch.tensor of size (1, n_latent)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: torch.tensor of size (n_documents,)
    """
    return -torch.cdist(M, v).squeeze()

def return_ranked(query, tokenizer, model, M):
    """
    Embed a `query` using `model` and `tokenizer`, and return the
    indices of document embeddings `M` sorted most to least similar.

    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: a list of ints of length `n_documents`
    """
    q = embed(query, tokenizer, model)
    sims = cosine_similarity(q, M)
    rankings = torch.argsort(sims, descending=True)
    sims = sims[rankings].tolist()
    ranks = rankings.tolist()
    return list(zip(ranks, sims))

def return_ranked_by_sentence(query, tokenizer, model, indices, M):
    """
    Embed a `query` using `model` and `tokenizer`, and return the
    indices of sentence embeddings `M` sorted by most to least similar.

    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    :param indices: torch.tensor of size (n_sentences)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: a list of ints of length `n_documents`
    """
    q = embed(query, tokenizer, model)
    sims = cosine_similarity(q, M)
    rankings = torch.argsort(sims, descending=True)
    doc_rankings = indices[rankings].numpy()
    _, first_doc_rankings = np.unique(doc_rankings, return_index=True)
    final_doc_rankings = doc_rankings[first_doc_rankings]
    matching_sims = sims[doc_rankings][first_doc_rankings]
    ranks = final_doc_rankings.tolist()
    matching_sims = matching_sims.tolist()
    return list(zip(ranks, matching_sims))