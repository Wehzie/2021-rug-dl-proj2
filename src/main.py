import enum
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data import Daily_Dialogue
from model_discriminator import Discriminator
from model_generator import Generator

# set seed
random.seed(24)

# Sets device to GPU preferred to CPU depending on what is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def trainG(batch_size, label, fake_label, netG):
    '''Train the Generator.'''
    
    noise = torch.randn(batch_size, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    return fake

def train(data_loader):
    '''Train generative adversarial network (GAN).'''
    
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

            print(f"real_cpu shape: {real_cpu.shape}")
            output = netD(real_cpu[0])  # only one 3-d vector is returned so removed 4th dimension

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
    print(f"Number of conversations: {len(data_loader)}")
    print(f"Dimensions of first conversation (vectorized): {data[0].size()}")
    train(data_loader)

if __name__ == "__main__":
    main()

# TODO: FIND THE WAY to do RNN without the stupid module which fucks it all up.
# maybe this https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch