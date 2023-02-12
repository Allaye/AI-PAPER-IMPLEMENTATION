import torch
import torch.nn as nn
import numpy as np


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # add the input and hidden layer together and pass it through the linear layer of both hidden and output
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # pass the output through the softmax function
        output = self.softmax(output)
        # return the output and hidden state output
        return output, hidden

    def init_hidden(self):
        # initialize the hidden state with zeros
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(50, n_hidden, 10)
print(f'rnn', rnn)
