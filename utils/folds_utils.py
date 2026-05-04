import json

def get_folds(cancer_name, src ="data/splits.json"):
    with open(src) as f:
        splits = json.load(f)
    data = splits[cancer_name]

    train = [[i - 1 for i in fold] for fold in data["train"]]
    test = [[i - 1 for i in fold] for fold in data["test"]]

    return train, test