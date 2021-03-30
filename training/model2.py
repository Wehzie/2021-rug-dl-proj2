from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os
import math


DATA_PATH = "/home/gery/Documents/Deep learning/I17-1099.Datasets/EMNLP_dataset/dialogues_text.txt"

qa_pairs = []

with open(DATA_PATH, "r") as f:
    for line in f:
        sentences = line.split("__eou__")
        sentences.pop()                                # remove the end-line
        for idx in range(len(sentences)-1):
            inputLine = sentences[idx].strip()            # .strip() removes start/end whitespace
            targetLine = sentences[idx+1].strip()

            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

with open('pairs.txt', 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=)
