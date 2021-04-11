import os
import pickle

import numpy as np


def get_embeddings():
    glove_path = os.path.join("data_app/generator", "glove")
    words = []
    idx = 0
    word2idx = {}

    with open(f"{glove_path}.txt", "rb") as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1

    pickle.dump(words, open(f"{glove_path}_words.pkl", "wb"))
    pickle.dump(word2idx, open(f"{glove_path}_idx.pkl", "wb"))
