# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import to_cuda


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1, dropout = 0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = 1)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, dropout = dropout)

    def forward(self, batch_src):
        embedded = self.embedding(batch_src)      # size: seq_len, batch_size, embed_size
        
        output, hidden = self.lstm(embedded)      # output size: seq_len, batch_size, hidden_size
        return output, hidden                     # hidden (tuple of hidden and cell of last time)
                                                  # hidden[0] size: num_layers, batch_size, hidden_size


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1, dropout = 0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = 1)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers = num_layers, dropout = dropout)
        
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        
    def forward(self, batch_trg, hidden):
        embedded = self.embedding(batch_trg)     # size: seq_len, batch_size, embed_size
        embedded = F.relu(embedded)
        
        output, hidden = self.lstm(embedded, hidden)
        
        output = self.out(output)                # size: seq_len, batch_size, vocab_size
        output = self.logsoftmax(output)
        
        return output[:-1]                   # size: seq_len-1 * batch_size * vocab_size

    def predict(self, word_id, hidden):      # predict next word given the current word id
        embedded = self.embedding(word_id)
        embedded = F.relu(embedded)
        output, hidden = self.lstm(embedded.view(1,1,-1), hidden)
        output = self.out(output)
        output = self.logsoftmax(output)

        return output.squeeze(), hidden             


class DecoderAttnRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1, dropout = 0):
        super(DecoderAttnRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = 1)
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_layers = num_layers, dropout = dropout)
        self.attn_combine = nn.Linear(embed_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        
    def forward(self, batch_trg, hidden_init, encoder_outputs, ifcuda):
        embedded = self.embedding(batch_trg)
        embedded = F.relu(embedded)
        
        output = Variable(torch.Tensor(1, batch_trg.size(1), self.vocab_size))
        output = to_cuda(output, ifcuda)
        
        decoder_hidden = hidden_init       # size: num_layers * batch_size * hidden_size 
        for t in range(batch_trg.size(0) - 1):
            # take out the hidden vector of the top layer
            hidden_top = decoder_hidden[-1]             # size: batch_size * hidden_size
            
            # calculating attention distribution
            attn_dist = [torch.bmm(encoder_outputs[k,:,:].unsqueeze(1),
                                   hidden_top.unsqueeze(2)).squeeze(2) 
                        for k in range(encoder_outputs.size(0))]
            attn_dist = torch.cat(attn_dist, dim = 1)   # size: batch_size * seq_len (src)
            attn_dist = F.softmax(attn_dist, dim = 1)
            
            # convex combination of encoder hiddens vectors based on attention
            hidden_new = torch.bmm(encoder_outputs.transpose(0,2).transpose(0,1),
                                   attn_dist.unsqueeze(2))
            hidden_new = hidden_new.squeeze()           # size: batch_size * hidden_size
            
            # combining word embedding and the context vector
            input = self.attn_combine(torch.cat([embedded[t], hidden_new], dim = 1))
            decoder_output, decoder_hidden = self.gru(input.unsqueeze(0), decoder_hidden)
            
            # calculating output distribution
            decoder_output = self.out(F.relu(decoder_output))
            output = torch.cat([output, decoder_output])
            
        output = self.logsoftmax(output[1:])
            
        return output                         # size: seq_len-1 * batch_size * vocab_size
    
    def predict(self, word_id, hidden, encoder_outputs):
                                              # predict next word given the current word_id
        embedded = self.embedding(word_id)
        embedded = F.relu(embedded)
        
        hidden_top = hidden[-1]               # size: 1 * hidden_size
                                     # encoder_outputs with size: seq_len * 1 * hidden_size
        attn_dist = torch.matmul(encoder_outputs.squeeze(), hidden_top.squeeze())
                                              # size: seq_len (src)
        attn_dist = F.softmax(attn_dist, dim = 0)
        
        context = torch.matmul(encoder_outputs.squeeze().t(), attn_dist)
        input = self.attn_combine(torch.cat([embedded.view(1, -1), context.view(1, -1)],
                                             dim = 1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.out(F.relu(output))
        output = self.logsoftmax(output)
        
        return output.squeeze(), hidden, attn_dist
