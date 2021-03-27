from operator import getitem
import random
from pathlib import Path
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim

# save/load data
import os
import json
import pickle
import lmdb

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

nltk.download('punkt')

# set seed
random.seed(24)

# save string data
def save_str_dat(data_path, data):
    #os.makedirs(data_path, exist_ok=True)
    with open(data_path, 'w') as file:
        json.dump(data, file, indent=1)

# save tensor data after vectorizing the strings
def save_vec_dat(data_path, data):
    #os.makedirs(data_path, exist_ok=True)
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)

# load data set
class DailyDialogue(Dataset):
    '''Daily Dialogue Dataset.'''

    def __init__(self):

        def get_str_dat():
            str_dat_path = Path("data/tokenized_str_dat.json")
            if str_dat_path.is_file():
                with open(str_dat_path, 'r') as file:
                    return json.load(file)
            
            # shape is 1 x number of conversations
            str_dat = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
            str_dat = str_dat[:10] # NOTE: testing
            
            # tokenize each conversation
            str_dat = [word_tokenize(conv.lower()) for conv in str_dat]

            # end of conversations indicated by "__eoc__" End-Of-Conversation token
            for conv in str_dat:
                conv[-1] = '__eoc__'
            
            save_str_dat(str_dat_path, str_dat)
            return str_dat
            
        def get_vec_dat(str_dat):
            vec_dat_path = Path("data/tokenized_vec_dat.json")
            if vec_dat_path.is_file():
                with open(vec_dat_path, 'rb') as file:
                    return pickle.load(file)

            model = gensim.models.Word2Vec(str_dat, size = 100, sg = 1, min_count = 1)
            print(model)

            vec_dat = []
            for conv in str_dat:
                temp_conversation = []
                for token in conv:
                    temp_conversation.append(model.wv[token,])
                vec = torch.FloatTensor(temp_conversation)
                vec_dat.append(vec)

            save_vec_dat(vec_dat_path, vec_dat)
            return vec_dat

        self.string_data = get_str_dat()
        self.vector_data = get_vec_dat(self.string_data)
        self.nr_of_samples = len(self.string_data)
    
    def __getitem__(self, index):
        return self.vector_data[index]
    
    def __len__(self):
        return self.nr_of_samples


# Sets device to GPU preferred to CPU depending on what is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            NotImplemented
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.a = nn.RNN(
            input_size = 100,  # size of a token vector
            hidden_size = 100,
            num_layers = 10,
            nonlinearity = 'tanh',
            bias = True,
            batch_first = False
        )
        # state size. 100 x 1
        self.b = nn.Linear(100, 1)
        # state size. 1 x 1
        self. c = nn.Sigmoid()
        # self.main = nn.Sequential(
        #     nn.RNN(
        #        input_size = 100,  # size of a token vector
        #        hidden_size = 100,
        #        num_layers = 10,
        #        nonlinearity = 'tanh',
        #        bias = True,
        #        batch_first = False
        #     ),
        #     # state size. 100 x 1
        #     nn.Linear(100, 1),
        #     # state size. 1 x 1
        #     nn.Sigmoid()
        #     # output is 1 x 1
        # )
    
    

    def unpack_sequence(packed_sequence, lengths):
        assert isinstance(packed_sequence, rnnutils.PackedSequence)
        head = 0
        trailing_dims = packed_sequence.data.shape[1:]
        unpacked_sequence = [torch.zeros(l, *trailing_dims) for l in lengths]
        # l_idx - goes from 0 - maxLen-1
        for l_idx, b_size in enumerate(packed_sequence.batch_sizes):
            for b_idx in range(b_size):
                unpacked_sequence[b_idx][l_idx] = packed_sequence.data[head]
                head += 1
        return unpacked_sequence

    # https://github.com/pytorch/examples/blob/master/dcgan/main.py
    def forward(self, input):
        # print(type(input))
        # output = self.main(input)
        x = self.a(input)   # tuple of tensors

        
        #If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
        c = self.unpack_sequence(x)
        print(type(c))
        x = self.b(c)
        x = self.c(x)
        output = x
        return output.view(-1, 1).squeeze(1)

def trainG(batch_size, label, fake_label, netG):
    # train with fake
    noise = torch.randn(batch_size, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    return fake

def train(data_loader):
    learning_rate = 0.05
    betas = (0.9, 0.999)            # first and second momentum
    epochs = 100
    
    netD = Discriminator().to(device)
    criterion = nn.BCELoss()        # Binary Cross Entropy loss

    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=betas)

    real_label = 1
    fake_label = 0

    # an epoch is a full iteration over the dataset
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for i, data in enumerate(data_loader):    # data is one conversation
            print(f"i {i}")
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = torch.FloatTensor(data).to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((len(data[0]),), real_label, dtype=real_cpu.dtype, device=device)

            # (13, 1, 100)
            print(f"len real_cpu {len(real_cpu)}")
            for token in real_cpu:
                output = netD(token)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            
            fake = trainG(batch_size, label, fake_label, NotImplemented)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

        torch.save(netD.state_dict(), './data/pytorch_out/netD_epoch_%d.pth' % (epoch))

def main():
    data = DailyDialogue()
    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=0)
    #train(data_loader)

    print(f"Number of Conversations: {len(data_loader)}")

    for i, j in enumerate(data_loader):
        print(i)
        print(j.size())
        print(j)
        print("")
        if i == 3: break

main()

# TODO: FIND THE WAY to do RNN without the stupid module which fucks it all up.
# maybe this https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch