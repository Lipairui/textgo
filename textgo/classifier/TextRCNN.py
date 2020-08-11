# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.maxpool = nn.MaxPool1d(args['max_len'])
        self.fc = nn.Linear(args['hidden_size'] * 2 + args['embed'], args['num_classes'])

    def forward(self, x):
        # x[0]: [batch_size, seq_len]
        # x[1]: [batch_size] actual len of each sentence
        emb = self.embedding(x[0]) # [batch_size, seq_len, embed]
        out, _ = self.lstm(emb) # [batch_size, seq_len, hidden_size*2] 
        out = torch.cat((emb, out), 2) # [batch_size, seq_len, hidden_size*2+embed]
        out = F.relu(out) # [batch_size, seq_len, hidden_size*2+embed]
        out = out.permute(0,2,1) # [batch_size, hidden_size*2+embed, seq_len]
        out = self.maxpool(out).squeeze() # [batch_size, hidden_size*2+embed]
        out = self.fc(out) # [batch_size, num_classes]
        return out


