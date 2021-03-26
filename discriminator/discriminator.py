import os
import random
from pathlib import Path
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import json
import lmdb

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
    data_path = Path("data/tokenized.s")
    os.makedirs(data_path, exist_ok=True)
    with open(data_path, 'w') as file:
        file.write(json.dumps(data))

# Load Data
def load_data():
    data = Path("data/tokenized")
    
    if data.is_file(): return json.loads(data)
    
    # shape is 1 x number of conversations
    data = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
    num_conversations = data.shape[0]
    data = data[:10] # NOTE: testing
    
    # tokenize each conversation
    data = [word_tokenize(conv.lower()) for conv in data]

    # end of conversations indicated by "__eoc__" End-Of-Conversation token
    for conv in data:
        conv[-1] = '__eoc__'

    model = gensim.models.Word2Vec(data, size = 100, sg = 1, min_count = 1)
    print(model)

    vectors = []
    for conv in data:
        temp_conversation = []
        for token in conv:
            temp_conversation.append(model.wv[token])
        vectors.append(temp_conversation)


    #save_vectors(vectors)
    #save_data(data)

    return vectors, data

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
            nn.RNN(
                input_size = 100,  # size of a token vector
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

def train(vectors):
    learning_rate = 0.05
    betas = (0.9, 0.999)    # first and second momentum
    epochs = 100

    netD = Discriminator().to(device)
    # Binary Cross Entropy loss
    criterion = nn.BCELoss()

    # load data
    train_data = vectors[0:-1000]
    test_data = vectors[-1000:]

    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=betas)

    # an epoch is a full iteration over the dataset
    for epoch in range(epochs):
        for i, data in enumerate(train_data, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = torch.from_numpy(data[0]).to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((1,), real_label, dtype=real_cpu.dtype, device=device)

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

        #torch.save(netD.state_dict(), '/netD_epoch_%d.pth' % (epoch))

def main():
    vectors, data = load_data()
    train(vectors)

main()

# TODO:
# use padding at end of conversation
# each token in a conversation is vectorized
# the discriminator takes as input a full conversation 
# the discriminator outputs a range from 0 to 1 indicating confidence that the conversation is authentic