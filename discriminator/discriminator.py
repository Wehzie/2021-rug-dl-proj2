from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Make sure we have consistent seeds.
random.seed(24)

# Load Data
def load_data():
    train_loader = torch.utils.data.DataLoader(NotImplemented)
    test_loader = torch.utils.data.DataLoader(NotImplemented)
    return train_loader, test_loader

####################
'''
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

What encoder/decoder do we use?

u_i = (x1 x2 x3 x4 .... xn-1 xn)
    where x_n is a word
    where u_i is an utterance (sentence)

Integer Encoding: Where each unique label is mapped to an integer.
One Hot Encoding: Where each label is mapped to a binary vector.
Learned Embedding: Where a distributed representation of the categories is learned.
Transformer: Meaningful embeddings
    u_i = yes -- transform --> [0, 0, 0]
    u_i = no -- transform --> [1, 1, 1]


https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

'''
####################

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

def main():
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
