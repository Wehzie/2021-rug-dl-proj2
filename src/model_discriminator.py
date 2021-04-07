import torch
import torch.nn as nn

class Discriminator(nn.Module):
    '''Discriminator module with recurrent layer.'''

    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_layers = 15
        self.rnn = nn.RNN(
            input_size = 100,       # size of a token vector
            hidden_size = 100,
            num_layers = hidden_layers,
            nonlinearity = 'tanh',  # TODO: try relu
        )
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.linear5 = nn.Linear(100, 100)
        self.linear6 = nn.Linear(100, 100)
        self.linear7 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.h0 = nn.parameter.Parameter(torch.randn(hidden_layers, 1, 100).type(torch.FloatTensor), requires_grad=True)

    # called when input is provided to the model
    def forward(self, input):
        x, hidden = self.rnn(input, self.h0)
        # x = self.linear1(input)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)
        # x = self.linear3(x)
        # x = self.relu(x)
        # x = self.linear4(x)
        # x = self.relu(x)
        # x = self.linear5(x)
        # x = self.relu(x)
        # x = self.linear6(x)
        # x = self.relu(x)
        # x, hidden = self.rnn(x, self.h0)
        x = self.linear7(x)
        output = self.sigmoid(x)
        return output.view(-1, 1).squeeze(1)