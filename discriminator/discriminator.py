from operator import getitem
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
import torch.nn.utils.rnn as rnnutils
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

class DailyDialogue(Dataset):
    def __init__(self):
        self.data_path = Path("data/tokenized.json")
        if self.data_path.is_file(): return json.loads(self.data)
        
        
        # shape is 1 x number of conversations
        self.data = np.loadtxt('./EMNLP_dataset/dialogues_text.txt', delimiter='\n', dtype=np.str, encoding='utf-8')
        self.data = self.data[:10] # NOTE: testing
        self.nr_of_samples = self.data.shape[0]
        
        # tokenize each conversation
        self.data = [word_tokenize(conv.lower()) for conv in self.data]

        # end of conversations indicated by "__eoc__" End-Of-Conversation token
        for conv in self.data:
            conv[-1] = '__eoc__'

        model = gensim.models.Word2Vec(self.data, size = 100, sg = 1, min_count = 1)
        print(model)

        vectors = []
        for conv in self.data:
            temp_conversation = []
            for token in conv:
                temp_conversation.append(model.wv[token,])
            vec = torch.FloatTensor(temp_conversation)
            vectors.append(vec)

        #save_vectors(vectors)
        #save_data(data)
        self.string_data = self.data
        self.data = vectors
    
    def __getitem__(self, index):
        return self.data[index]
    
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
    betas = (0.9, 0.999)    # first and second momentum
    epochs = 100

    netD = Discriminator().to(device)
    # Binary Cross Entropy loss
    criterion = nn.BCELoss()

    # load data
    # train_data = vectors[0:-5] # -1000
    # test_data = vectors[-5:] # -1000


    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=betas)

    # an epoch is a full iteration over the dataset
    for epoch in range(epochs):
        print(f"epoch {epoch}")
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
    train(data_loader)

main()

# TODO: FIND THE WAY to do RNN without the stupid module which fucks it all up.
# maybe this https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch