import os
import random
from pathlib import Path
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


nltk.download('punkt')


# Make sure we have consistent seeds.
random.seed(24)

def save_data(data):
    NotImplemented

# Load Data
def load_data():
    data = Path("/data/tokenized")
    
    if data.is_file(): return data
    
    # shape is 1 x number of conversations
    data = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
    num_conversations = data.shape[0]
    
    # split each conversation into n utterances
    data = [conv.split('__eou__') for conv in data]
    
    # split each utterance in a conversation into n words
    data = [[word_tokenize(utter.lower()) for utter in conv] for conv in data]

    print(len(data))

    #save_data(data)

    temp_vocab = []
    for conv in data:
        for utter in conv:
            temp_vocab.append(utter)

    print(len(temp_vocab))
    model = gensim.models.Word2Vec(temp_vocab, size = 100, sg = 1, min_count = 1)

    vectors = []
    for i in range(0, len(data)):
        temp_sentence = []
        for j in range(0, len(data[i])):
            temp_word = []
            for k in range(0, len(data[i][j])):
                temp_word.append(model.wv[data[i][j][k]])
            temp_sentence.append(temp_word)
        vectors.append(temp_sentence)

    data = vectors
    return data

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
        self.main = nn.Sequential(
            # input is 64 x 64
            nn.RNN(
                input_size = [64, 64],
                hidden_size = 100,
                num_layers = 10,
                nonlinearity = 'tanh',
                bias = True,
                batch_first = False
            ),
            # state size. 100 x 1
            nn.Linear(100, 1),
            # state size. 1 x 1
            nn.Sigmoid()
            # output is 1 x 1
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def trainG(batch_size, label, fake_label, netG):
    # train with fake
    noise = torch.randn(batch_size, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    return fake

def train():
    learning_rate = 0.05
    betas = (0.9, 0.999)    # first and second momentum
    epochs = 100

    netD = Discriminator().to(device)
    # Binary Cross Entropy loss
    criterion = nn.BCELoss()

    # load data
    train_data, test_data = load_data()

    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=betas)

    for epoch in range(epochs):
        for i, data in enumerate(train_data, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
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

        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (epoch))

def main():
    data = load_data()
    print(data[0][0][0])

main()