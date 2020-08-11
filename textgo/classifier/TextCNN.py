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
        self.convs = nn.ModuleList([nn.Conv2d(1, args['num_filters'], (k, args['embed'])) for k in args['filter_sizes']])
        self.dropout = nn.Dropout(args['dropout'])
        self.fc = nn.Linear(args['num_filters'] * len(args['filter_sizes']), args['num_classes'])

    def conv_and_pool(self, x, conv):
        # x: [batch_size, 1, seq_len, embed]
        x = F.relu(conv(x)).squeeze(3) # [batch_size, num_filters, features]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [batch_size, num_filters]
        return x

    def forward(self, x):
        # x[0]: [batch_size, seq_len]
        out = self.embedding(x[0]) # [batch_size, seq_len, embed]
        out = out.unsqueeze(1) # [batch_size, 1, seq_len, embed]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # [batch_size, num_filters*len(filter_sizes)]
        out = self.dropout(out) # [batch_size, num_filters*len(filter_sizes)]
        out = self.fc(out) # [batch_size, num_classes]
        return out


