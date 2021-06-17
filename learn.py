import json
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC

from search import *


def get_label(index, metadata):
    title = metadata[str(index)]['project_title']
    input_string = f"""Please label:
    {"-"*len(title)}
    {title}
    {"-"*len(title)}
    """
    label = ""
    while label.lower() not in ["t", "f", "y", "n"]:
        label = input(input_string)
    if label.lower in ["t", "y"]:
        label = 1
    elif label.lower in ["f", "n"]:
        label = 0
    return label

def active_learn(X, metadata, n_samples=5, n_rounds=5, prelearn=None):
    learner = ActiveLearner(estimator=GaussianNB(),
                            query_strategy=uncertainty_sampling)
    # pre-train with most relevant labels
    if prelearn is not None:
        prelearn_labels = []
        for idx in prelearn:
            prelearn_labels.append(get_label(idx, metadata))
        learner.teach(X[prelearn], prelearn_labels)
    
    # then run regular training
    for _ in range(n_rounds):
        # TODO: why does this always choose stenting / papers starting with P?
        request, _ = learner.query(X, n_instances=n_samples)
        labels = []
        for index in request:
            labels.append(get_label(index, metadata))
        labels = np.array(labels, dtype=bool)
        learner.teach(X[request], labels)
    probabilities = learner.predict_proba(X)[:, 1]
    return probabilities

def main():
    # load metadata
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained("./model/distilbert3")
    model = AutoModelForMaskedLM.from_pretrained("./model/distilbert3")

    # load embeddings
    M = torch.load("./data/distilbert3tensor.pt")

    # get most relevant
    query = input("Query: ")

    # get top list
    top_idx = return_ranked(query, tokenizer, model, M)
    prelearn = [el[0] for el in top_idx
                if len(metadata[str(el[0])]['abstract'].split()) > 200][:30]  # caution: use minwords
    prelearn = np.random.choice(prelearn, size=10, replace=False)
    print(f"Prelearn: {prelearn}")

    # learn
    M = M.detach().numpy()
    probs = active_learn(M, metadata, n_samples=5, n_rounds=3, prelearn=prelearn)
    top_probs = np.argsort(probs)[::-1]
    top_probs = [el for el in top_probs
                if len(metadata[str(el)]['abstract'].split()) > 200]  # caution: use minwords
    top_probs = top_probs[:50]
    announce_str = "Your top recommendations are:"
    print("\n\n")
    print(announce_str)
    print("-"*len(announce_str))
    for idx in top_probs:
        print(metadata[str(idx)]['project_title'])

if __name__ == "__main__":
    main()

