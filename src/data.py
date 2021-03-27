import sys
import os
from pathlib import Path
import pickle
import json

import numpy as np
import gensim
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import torch

from torch.utils.data import Dataset

# load data set
class Daily_Dialogue(Dataset):
    '''Daily Dialogue Dataset.'''

    def __init__(self):
        self.string_data = self.get_str_dat()
        self.vector_data = self.get_vec_dat(self.string_data)
        self.nr_of_samples = len(self.string_data)

    def get_str_dat(self):
        # try loading from file
        str_dat_path = Path("data/tokenized_str_dat.json")
        #if str_dat_path.is_file():
        #    with open(str_dat_path, 'r') as file:
        #        return json.load(file)
        
        # shape is 1 x number of conversations
        str_dat = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        str_dat = str_dat[:10] # NOTE: testing
        
        # tokenize each conversation
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]

        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            conv[-1] = '__eoc__'
        
        self.save_str_dat(str_dat_path, str_dat)
        return str_dat
        
    def get_vec_dat(self, str_dat):
        # try loading from file
        vec_dat_path = Path("data/tokenized_vec_dat.json")
        #if vec_dat_path.is_file():
        #    with open(vec_dat_path, 'rb') as file:
        #        return pickle.load(file)

        # initialize encoder decoder model
        model = gensim.models.Word2Vec(str_dat, size = 100, sg = 1, min_count = 1)
        print(model)

        vec_dat = []
        for conv in str_dat:
            temp_conversation = []
            for token in conv:
                temp_conversation.append(model.wv[token,])
            vec = torch.FloatTensor(temp_conversation)
            vec_dat.append(vec)

        self.save_vec_dat(vec_dat_path, vec_dat)
        return vec_dat

    # save string data
    def save_str_dat(self, data_path, data):
        #os.makedirs(data_path, exist_ok=True)
        with open(data_path, 'w') as file:
            json.dump(data, file, indent=1)

    # save tensor data after vectorizing the strings
    def save_vec_dat(self, data_path, data):
        #os.makedirs(data_path, exist_ok=True)
        with open(data_path, 'wb') as file:
            pickle.dump(data, file)
    
    def __getitem__(self, index):
        return self.vector_data[index]
    
    def __len__(self):
        return self.nr_of_samples

