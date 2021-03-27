import sys
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from daily_dialogue import Daily_Dialogue

# set seed
random.seed(24)

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
    data = Daily_Dialogue()
    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=0)
    #train(data_loader)

    print(f"Number of Conversations: {len(data_loader)}")

    for i, j in enumerate(data_loader):
        print(i)
        print(j.size())
        print(j)
        print("")
        if i == 3: break

if __name__ == "__main__":
    main()

# TODO: FIND THE WAY to do RNN without the stupid module which fucks it all up.
# maybe this https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch