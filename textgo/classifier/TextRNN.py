# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args['embedding_pretrained'] is not None:
            self.embedding = nn.Embedding.from_pretrained(args['embedding_pretrained'], freeze=False)
        else:
            self.embedding = nn.Embedding(args['n_vocab'], args['embed'], padding_idx=args['n_vocab'] - 1)
        self.lstm = nn.LSTM(args['embed'], args['hidden_size'], args['num_layers'],
                            bidirectional=True, batch_first=True, dropout=args['dropout'])
        self.fc = nn.Linear(args['hidden_size'] * 2, args['num_classes'])

    def forward(self, x):
        # x[0]: [batch_size, seq_len]
        # x[1]: [batch_size] actual len of each sentence
        emb = self.embedding(x[0]) 
        # emb: [batch_size, seq_len, embed]
        packed_input = pack_padded_sequence(emb, x[1], batch_first=True, enforce_sorted=False) # use this function to speed up
        packed_output, (hidden, cell) = self.lstm(packed_input) 
        # hidden: [num_directions, batch_size, hidden_size]
        out, seq_len = pad_packed_sequence(packed_output, batch_first=True) 
        # out: [batch_size, seq_len, hidden_size*num_directions]
        # Note: Do not use out[:,-1,:] as the target hidden state, since it's all zeros.
        out = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim = 1)
        # out: [batch_size, hidden_size*num_directions]
        out = self.fc(out) # [batch_size, num_classes]
        return out


