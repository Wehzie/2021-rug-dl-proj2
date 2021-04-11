import sys
import os
from pathlib import Path
import pickle
import json
import random

import numpy as np
import gensim
import nltk
from nltk.tokenize import word_tokenize
import torch

from torch.utils.data import Dataset

# load data set
class Daily_Dialogue(Dataset):
    '''Daily Dialogue Dataset.'''

    def __init__(self,train_mode):
        self.word_vector_size = 100
        self.max_conv_len = 0
        self.model = self.make_or_load_model()
        # self.model.save('./Discriminator/gensim_model.model')
        if train_mode:
            # get true data

            self.string_data = self.get_str_dat()
            self.nr_of_true_samples = len(self.string_data)

            # get labels for true conversations
            self.target = [1 for i in range(self.nr_of_true_samples)]

            # get fake data
            self.string_data = self.string_data + self.get_random_dat() # NOTE: comment for testing
            # self.string_data = self.string_data + self.get_str_dat(False) 
            self.nr_of_samples = len(self.string_data)

            # get labels for false conversations
            self.target = self.target + [0 for i in range(self.nr_of_true_samples, self.nr_of_samples)]

            self.vector_data = self.get_vec_dat(self.string_data)

        else:
            self.string_data = self.get_final_data()
            self.nr_of_true_samples = len(self.string_data)

            # get labels for true conversations
            self.target = [1 for i in range(self.nr_of_true_samples)]

            # get fake data
            self.string_data = self.string_data + self.get_generator_dat() # NOTE: comment for testing
            # self.string_data = self.string_data + self.get_str_dat(False) 
            self.nr_of_samples = len(self.string_data)

            # get labels for false conversations
            self.target = self.target + [0 for i in range(self.nr_of_true_samples, self.nr_of_samples)]

            self.vector_data = self.get_vec_dat(self.string_data)
        
    def make_or_load_model(self):
        try:
            print("old")
            model = gensim.models.Word2Vec.load('./Discriminator/gensim_model.model')
        except:
            print("new")
            str_dat = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
            # tokenize each conversation
            str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
        
            # end of conversations indicated by "__eoc__" End-Of-Conversation token
            for conv in str_dat:
                if len(conv) > self.max_conv_len:
                    self.max_conv_len = len(conv)
                conv[-1] = '__eoc__'
            print(self.max_conv_len)

            model = gensim.models.Word2Vec(str_dat, size = self.word_vector_size, sg = 1, min_count = 1)
        print(model)
        return model
                       
    # shape is 1 x number of conversations
    def get_str_dat(self):
        str_dat = np.loadtxt('./EMNLP_dataset/train/dialogues_train.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
    
        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            conv[-1] = '__eoc__'

        return str_dat
        
    def get_random_dat(self):
        str_dat = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        str_dat = [conv.split('__eou__') for conv in str_dat]

        temp_str_dat = []
        for conv in str_dat:
            temp_conv = []
            for utter in conv:
                if utter != ' ' and utter != '':
                    temp_utter = utter
                    temp_conv.append(temp_utter)
            temp_str_dat.append(temp_conv)
        str_dat = temp_str_dat

        temp_dat = []
        lengths =[]
        for conv in str_dat:
            lengths.append(len(conv))
            temp_dat = temp_dat + conv

        str_dat = temp_dat
        random.shuffle(str_dat)

        temp_str_dat = []
        for i in lengths:
            temp_conv = ''
            for j in range(i):
                temp_utter = random.choice(str_dat)
                while temp_utter == '' or temp_utter == ' ' or temp_utter == '  ':
                    temp_utter = random.choice(str_dat)
                temp_conv = temp_conv + temp_utter + ' __eou__ '
            temp_conv = temp_conv[:-9]
            temp_str_dat.append(temp_conv)

        str_dat = temp_str_dat
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]

        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            if len(conv) > self.max_conv_len:
                self.max_conv_len = len(conv)
            conv[-1] = '__eoc__'
        print(self.max_conv_len)

        return str_dat

    def get_vec_dat(self, str_dat):
        vec_dat = []

        for conv in str_dat:
            temp_conversation = []
            for token in conv:
                try:
                    temp_conversation.append(self.model.wv[token,])
                except:
                    print("Not in vocabulary: ")
                    print(token)
            for i in range(875 - len(conv)):
                pad = np.zeros((1, self.word_vector_size))
                temp_conversation.append(pad)
            vec = torch.FloatTensor(temp_conversation)
            vec_dat.append(vec)
        
        return vec_dat

    def get_final_data(self):
        str_dat = np.loadtxt('./EMNLP_dataset/test/dialogues_test.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
    
        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            conv[-1] = '__eoc__'
        
        return str_dat

    def get_generator_dat(self):
        str_dat = np.loadtxt('./Generator/fake conversations/conversations_2.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        # str_dat = np.loadtxt('./Generator/fake conversations/conversations_3.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        # str_dat = np.loadtxt('./Generator/fake conversations/conversation_glove_dot.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        # str_dat = np.loadtxt('./Generator/fake conversations/conversation_glove_general.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
    
        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            conv[-1] = '__eoc__'
        
        return str_dat

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
        return self.vector_data[index], self.target[index]
    
    def __len__(self):
        return len(self.vector_data)


