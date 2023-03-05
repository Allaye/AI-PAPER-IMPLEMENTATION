import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloader import vocab_transform


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, drop_prob=0.2):
        super(Encoder, self).__init__()
        # initialize the hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_prob)

        # initialize the learning layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # embedding layer to convert the input to a vector
        # of meaningful representation
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop_prob)  # lstm layer for the encoding

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))  # pass the input through the embedding layer
        outputs, (hidden, cell) = self.lstm(embedding)  # pass the embedding through the lstm layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, out_put, drop_prob=0.2):
        super(Decoder, self).__init__()
        # initialize the hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_prob)

        # initialize the learning layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # embedding layer to convert the input to a vector
        # of meaningful representation
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop_prob)  # lstm layer for the encoding
        self.fc = nn.Linear(hidden_size, out_put)  # fully connected layer to convert the output to the desired output

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)  # reshape the input
        embedding = self.dropout(self.embedding(x))  # pass the input through the embedding layer
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))  # pass the embedding through the lstm layer
        predictions = self.fc(outputs).squeeze(0)  # pass the output through the fully connected layer
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # initialize the building blocks of the seq2seq model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(vocab_transform['en'])

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs
