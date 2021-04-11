import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator module with recurrent layer."""

    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_layers = 15
        self.rnn = nn.RNN(
            input_size=100,  # size of a token vector
            hidden_size=100,
            num_layers=hidden_layers,
            nonlinearity="tanh",
        )
        max_conv_len = 875
        self.linear1 = nn.Linear(max_conv_len*100, 1000)
        self.linear2 = nn.Linear(1000, 400)
        self.linear3 = nn.Linear(400, 100)
        self.flatter = nn.Flatten(start_dim=0, end_dim=-1)

        self.linear = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.h0 = nn.parameter.Parameter(
            torch.randn(hidden_layers, 1, 100).type(torch.FloatTensor),
            requires_grad=True,
        )

    # called when input is provided to the model
    def forward(self, input):
        x = self.flatter(input)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear(x)
        output = self.sigmoid(x)

        return output.view(-1, 1).squeeze(1)