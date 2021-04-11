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
from torch.nn.parameter import Parameter

from data import Daily_Dialogue
from model_discriminator import Discriminator
from model_generator import Generator

# set seed
random.seed(24)

# Sets device to GPU preferred to CPU depending on what is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_mode = False

def train(data):
    '''Train generative adversarial network (GAN).'''
    split = 0.8
    train_len = int(len(data)*split)
    test_len = len(data) - int(len(data)*split)
    data_train, data_test = torch.utils.data.random_split(data, [train_len,test_len])
    data_loader = DataLoader(dataset=data_train, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0)
    print(f"Number of conversations: {len(data_loader)}")

    learning_rate = 0.001
    betas = (0.9, 0.999)            # first and second momentum
    epochs = 20

    netD = Discriminator().to(device)
    criterion = nn.BCELoss()        # Binary Cross Entropy loss
    optimizerD = optim.SGD(netD.parameters(), lr=learning_rate)

    real_label = 1
    fake_label = 0

    count = 0
    
    for epoch in range(epochs):  # an epoch is a full iteration over the dataset
        print(f"Epoch: {epoch}")
        for i, (conv, label) in enumerate(data_loader):  # conv is one conversation
            
            ############################
            # (1) Update D network.
            ###########################
            netD.zero_grad()  # initialize gradients with zero
            real_cpu = conv.to(device)  # transfer tensor to CPU
            batch_size = real_cpu.size(0)  # batch size is number of conversations (1) handled per iteration

            output = netD(real_cpu[0])

            # Tensor magic to only look at the last label and last output
            label = torch.FloatTensor([label]).to(device)
            out = output

            errD_real = criterion(out, label)
            errD_real.backward()
            D_x = output.mean().item()
            optimizerD.step()
            count = count + 1
        test_model(test_loader,netD)
        # TODO fix saving the model
        torch.save(netD.state_dict(), './results/discriminator_model/netD_epoch_%d.pth' % (epoch))
        
def test_model(test_loader, netD):
    correct = 0
    total = 0
    false_neg = 0
    false_pos = 0
    top = 0
    bot = 1
    topcon = 0
    botcon = 0
    print("testing....")
    for i, (conv, label) in enumerate(test_loader):  # conv is one conversation
        real_cpu = conv.to(device)              # transfer tensor to CPU
        batch_size = real_cpu.size(0)           # batch size is number of conversations (1) handled per iteration
                                                #   size(0) takes first argument of tensor shape
        output = netD(real_cpu[0])
        if output[-1].item() > 0.5:
            classified = 1
        else:
            classified = 0
        
        if label - classified == 0:
            if label == 0:
                if output[-1].item() < bot and not train_mode:
                    bot = output[-1].item()
                    botcon = i
            correct = correct + 1
        else:
            if label - classified > 0.1:
                false_neg = false_neg + 1
                # print("false negative " + str(output[-1].item()))
            if label - classified < -0.1:
                false_pos = false_pos + 1
                if output[-1].item() > top and not train_mode:
                    top = output[-1].item()
                    topcon = i
                    print(f"Conversation: {i}")
                # print("false positive " + str(output[-1].item()))
            
        total = total + 1

    accuracy = correct/total
    print("accuracy " + str(accuracy))
    print("false negatives ratio " + str(false_neg/total))
    print("false positives ratio " + str(false_pos/total))

    if not train_mode:
        print("best: " + str(topcon))
        print("worst: " + str(botcon))

def main():
    data = Daily_Dialogue(train_mode)
    if train_mode:
        train(data)
    else:
        model = Discriminator().to(device)
        state = torch.load('./results/discriminator_model/netD_epoch_10.pth', map_location=torch.device('cuda'))
        model.load_state_dict(state)
        test_loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=0)
        test_model(test_loader, model)


if __name__ == "__main__":
    main()