import torch.nn as nn

class Discriminator(nn.Module):
    '''Discriminator module with recurrent layer.'''

    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn = nn.RNN(
            input_size = 100,       # size of a token vector
            hidden_size = 100,
            num_layers = 10,
            nonlinearity = 'tanh',
        )
        self.linear = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    # called when input is provided to the model
    def forward(self, input):
        x, hidden = self.rnn(input)
        x = self.linear(x)
        output = self.sigmoid(x)
        return output.view(-1, 1).squeeze(1)