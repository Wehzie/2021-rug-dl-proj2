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
# nltk.download('punkt')
import torch

from torch.utils.data import Dataset

# load data set
class Daily_Dialogue(Dataset):
    '''Daily Dialogue Dataset.'''

    def __init__(self,train_mode):
        self.word_vector_size = 100
        self.max_conv_len = 0
        self.model = self.make_or_load_model()
        # self.model.save('./src/gensim_model.model')
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
            model = gensim.models.Word2Vec.load('./src/gensim_model.model')
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
                       

    def get_str_dat(self):
        # try loading from file
        # str_dat_path = Path("data/tokenized_str_dat.json")
        #if str_dat_path.is_file():
        #    with open(str_dat_path, 'r') as file:
        #        return json.load(file)
        
        # shape is 1 x number of conversations
        

        str_dat = np.loadtxt('./EMNLP_dataset/train/dialogues_train.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
    
        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            conv[-1] = '__eoc__'

        # str_dat = str_dat[:3000] # NOTE: testing
        
        # if fake:
        #     fake_dat = np.loadtxt('./training/conversations.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        #     fake_dat = [word_tokenize(conv.lower()) for conv in fake_dat]
        #     for conv in fake_dat:
        #         conv[-1] = '__eoc__'
        #     str_dat = fake_dat
        # self.save_str_dat(str_dat_path, str_dat)
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



        # within sentence shuffle
        # for conv in str_dat:
        #     random.shuffle(conv)
        # temp_str_dat = []
        # for conv in str_dat:
        #     temp_conv = ''
        #     for sentence in conv:
        #         if sentence != ' ' and sentence != '':
        #             if temp_conv:
        #                 temp_conv = temp_conv + sentence + ' __eou__ '
        #             else:
        #                 temp_conv = sentence + ' __eou__ '

        #     temp_conv = temp_conv[:-9]
        #     temp_str_dat.append(temp_conv)
        # str_dat = temp_str_dat

        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]

        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            if len(conv) > self.max_conv_len:
                self.max_conv_len = len(conv)
            conv[-1] = '__eoc__'
        print(self.max_conv_len)

        # str_dat = str_dat[:3000] # NOTE: testing

        # self.save_str_dat(str_dat_path, str_dat)
        return str_dat

    def get_vec_dat(self, str_dat):
        # try loading from file
        # vec_dat_path = Path("data/tokenized_vec_dat.json")
        #if vec_dat_path.is_file():
        #    with open(vec_dat_path, 'rb') as file:
        #        return pickle.load(file)
        # TODO: if we save/load vectors we also want to save/load the model for decoding, use model.save()

        # initialize encoder decoder model
        
        # model = gensim.models.Word2Vec(str_dat, size = self.word_vector_size, sg = 1, min_count = 1)
        # self.model = model
        # print(model)

        vec_dat = []

        for conv in str_dat:
            temp_conversation = []
            for token in conv:
                try:
                    temp_conversation.append(self.model.wv[token,])
                except:
                    print("Not in vocabulary: ")
                    print(token)
                # print(model.wv[token,].shape)
            #for i in range(self.max_conv_len - len(conv)):
            for i in range(875 - len(conv)):
                pad = np.zeros((1, self.word_vector_size))
                # print(pad.shape)
                temp_conversation.append(pad)
            vec = torch.FloatTensor(temp_conversation)
            vec_dat.append(vec)


        # self.save_vec_dat(vec_dat_path, vec_dat)
        
        return vec_dat

    # is this possibe?
    # def decode(self, vec_dat: list) -> list:
    #     out = []
    #     for token in vec_dat:
    #         NotImplemented
    #     return out

    #save string data

    def get_final_data(self):
        # str_dat = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        # str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
    
        # # end of conversations indicated by "__eoc__" End-Of-Conversation token
        # for conv in str_dat:
        #     if len(conv) > self.max_conv_len:
        #         self.max_conv_len = len(conv)
        #     conv[-1] = '__eoc__'
        # print(self.max_conv_len)

        # model = gensim.models.Word2Vec(str_dat, size = self.word_vector_size, sg = 1, min_count = 1)
        # self.model = model
        # print(model)

        str_dat = np.loadtxt('./EMNLP_dataset/test/dialogues_test.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
        # tokenize each conversation
        str_dat = [word_tokenize(conv.lower()) for conv in str_dat]
    
        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in str_dat:
            conv[-1] = '__eoc__'
        
        return str_dat

    def get_generator_dat(self):
        str_dat = np.loadtxt('./training/conversations_2.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        
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


