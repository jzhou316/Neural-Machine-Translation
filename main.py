# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models import *
from utils import *
from beamsearch import *

torch.manual_seed(0)

### import data ###
batch_size = 64
train_iter, val_iter, DE, EN, train, val, test = loadIWSLT(batch_size)


### set parameters ###
DEvocab_size = len(DE.vocab)
ENvocab_size = len(EN.vocab)
embed_size = 200
hidden_size = 200
num_layers = 2
dropout = 0.3
ifcuda = True

lr = 0.005
num_epoch = 1


### without Attention ###
encoder = EncoderRNN(DEvocab_size, embed_size, hidden_size,
                     num_layers = num_layers, dropout = dropout)
decoder = DecoderRNN(ENvocab_size, embed_size, hidden_size,
                     num_layers = num_layers, dropout = dropout)
lossfn = nn.NLLLoss(size_average = True, ignore_index = -100)

# training
encoder, decoder = training(train_iter, encoder, decoder, lossfn,
                            lr, num_epoch, ifcuda = ifcuda)

# validating
lossfn_val = nn.NLLLoss(size_average = False, ignore_index = -100)
loss_avg = validating(val_iter, encoder, decoder, lossfn_val, ifcuda = ifcuda)

# beam search
val_iter.init_epoch()
K = 3
num_steps = 5
batch = next(iter(val_iter))
encoder.eval()
decoder.eval()
for i in range(batch.batch_size):
    batch.src = to_cuda(batch.src, ifcuda)
    encoder_outputs, hidden = encoder(batch.src[:,i].unsqueeze(1))
    beam = Beam(K, 1, 2, 3, hidden)
    for n in range(num_steps):
        beam.beamstep(decoder, ifcuda = ifcuda)
    seq, attn = beam.retrieve()
    print([EN.vocab.itos[seq[m]] for m in range(len(seq))])



### with Attention ###
encoderAttn = EncoderRNN(DEvocab_size, embed_size, hidden_size,
                         num_layers = num_layers, dropout = dropout)
decoderAttn = DecoderAttnRNN(ENvocab_size, embed_size, hidden_size,
                             num_layers = num_layers, dropout = dropout)
lossfnAttn = nn.NLLLoss(size_average = True, ignore_index = -100)

# training
encoderAttn, decoderAttn = trainingAttn(train_iter, encoderAttn, decoderAttn, lossfnAttn,
                                        lr, num_epoch, ifcuda = ifcuda)

# validating
lossfn_val = nn.NLLLoss(size_average = False, ignore_index = -100)
loss_avg = validatingAttn(val_iter, encoderAttn, decoderAttn, lossfn_val, ifcuda = ifcuda)

# beam search
val_iter.init_epoch()
K = 3
num_steps = 5
batch = next(iter(val_iter))
encoderAttn.eval()
decoderAttn.eval()
for i in range(batch.batch_size):
    batch.src = to_cuda(batch.src, ifcuda)
    encoder_outputs, hidden = encoderAttn(batch.src[:,i].unsqueeze(1))
    beam = Beam(K, 1, 2, 3, hidden[0])
    for n in range(num_steps):
        beam.beamstep(decoderAttn, encoder_outputs = encoder_outputs,
                               ifAttn = True, ifcuda = ifcuda)
    seq, attn = beam.retrieve()
    print([EN.vocab.itos[seq[m]] for m in range(len(seq))])


