import torch.nn as nn
import torch.nn.utils.rnn as rnnutils
import torch.optim as optim

class Discriminator(nn.Module):
    '''Discriminator module with recurrent layer.'''

    def __init__(self):
        super(Discriminator, self).__init__()
        self.a = nn.RNN(
            input_size = 100,  # size of a token vector
            hidden_size = 100,
            num_layers = 10,
            nonlinearity = 'tanh',
            bias = True,
            batch_first = False,
            bidirectional = True
        )
        # state size. 100 x 1
        self.b = nn.Linear(200, 1) # TODO: generalize 13, this is the number of tokens in the first conversation
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

    # https://github.com/pytorch/examples/blob/master/dcgan/main.py
    def forward(self, input):
        x, hidden = self.a(input)
        x = self.b(x)
        x = self.c(x)
        output = x
        return output.view(-1, 1).squeeze(1)