import torch
from absl import app
from absl import flags
from model import get_infra
from fetch import get_sentences

FLAGS = flags.FLAGS

flags.DEFINE_string("q", None, "The query (topic).")
flags.DEFINE_integer("n", 5, "The number of results to return.")
flags.DEFINE_bool("scibert", False, "If True, scibert model is used.")
flags.DEFINE_bool("train", False, "If True, model is fine-tuned first.")
flags.DEFINE_string("docpath", None, "The filepath to the structured abstracts .json file. (Soon database?)")

def mean_pooling(token_embeddings, attention_mask):
    """
    Effectively averages the embeddings of tokens across the vocabulary dimension
    to calculate the vocab-weighted latent representations (embeddings).

    :param token_embeddings: torch.float tensor of size (n_examples, n_vocab, n_latent)
    :param attention_mask: torch.byte tensor of size (n_examples, n_vocab)
    :return: torch.float tensor of size (n_examples, n_latent)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def similarity(v, M):
    """
    Cosine similarity between a vector (single query embedding) and a matrix (of embeddings).

    :param v: torch.tensor of size (1, n_latent)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: torch.tensor of size (n_documents,)
    """
    dv = v.norm(p=2)
    dM = M.norm(p=2, dim=1)
    return (M.matmul(v.T)).squeeze().div(dM * dv)

def search(model, tokenizer, query, latents, sentences, orig_mask, n_results=5):
    """
    Given a query, return the n most related documents/sentences.

    TODO: currently each sentence is a document. Switch to document -> [sentences]
          so that it returns document title and information, not the sentence.

    TODO: store latents separately to save on inference / stop recomputing each query

    :param model: a BERT model
    :param query: str of the search query
    :param latents: torch.tensor of previously-calculated model latents
    :param sentences: list of strings of previous sentences / abstracts
    :param orig_mask: torch.tensor of tokenizer attention mask
    :param n_results: int of number of top-match strings to return
    :return: list of strings
    """
    token = tokenizer([query], return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**token, output_hidden_states=True).hidden_states[-1]
    ## use pooling across vocab size
    latents = mean_pooling(latents, orig_mask)
    query_embedding = mean_pooling(query_embedding, token['attention_mask'])
    ## compute similarity
    sim = similarity(query_embedding, latents)  ## TODO: investigate alternative metrics
    rankings = torch.argsort(sim, descending=True)
    return [sentences[i] for i in rankings.tolist()[:n_results]]


def main(argv):
    assert (FLAGS.q is not None), "Query flag (-q) can't be empty."
    assert (not FLAGS.train), "Fine-tuning the model isn't implemented yet."
    assert (FLAGS.docpath is None), "Custom document path isn't supported yet."

    model, tokenizer = get_infra(FLAGS.scibert, FLAGS.train)
    if FLAGS.docpath is None:
        docs = get_sentences()
    doc_tokens = tokenizer(docs, return_tensors="pt", padding=True, truncation=True, max_length=192)
    latents = model(**doc_tokens, output_hidden_states=True).hidden_states[-1]
    output = search(model, tokenizer, FLAGS.q, latents, docs, doc_tokens['attention_mask'], FLAGS.n)
    header = "OUTPUTS:"
    print(header)
    print(len(header)*"-")
    print(*["\n" + f"{i}. {el}" for i, el in enumerate(output)])
    print(len(header)*"-")
    # return output


if __name__ == "__main__":
    app.run(main)