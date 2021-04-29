from transformers import AutoTokenizer, AutoModelForMaskedLM

def get_infra(scibert=False, train=False):
    """
    Returns a model and tokenizer.

    :param scibert: bool, returns scibert model if True.
    :param train: bool, returns trainable model if True.
    :return:
    """
    model_name = "distilbert-base-uncased" if not scibert else "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    if train:
        model.train()
    return model, tokenizer

def finetune(model, data):
    """
    Fine-tunes the model.

    :param model:
    :param data:
    :return: model
    """
    raise NotImplementedError