# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import spacy
import time
import math


def to_cuda(x, ifcuda):
    if torch.cuda.is_available() and ifcuda:
        x = x.cuda()
    return x


def timeSince(start):
    now = time.time()
    s = now - start
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def loadIWSLT(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=tokenize_de)
    EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

    MAX_LEN = 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)

    MIN_FREQ = 5
    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)

    BATCH_SIZE = batch_size
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                  repeat=False, sort_key=lambda x: len(x.src))
    print('Finished data loading!\n')

    return train_iter, val_iter, DE, EN, train, val, test


def training(train_iter, encoder, decoder, lossfn, lr, num_epoch, ifcuda = False):
    encoder = to_cuda(encoder, ifcuda)
    decoder = to_cuda(decoder, ifcuda)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr = lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = lr)
    start = time.time()
    encoder.train()
    decoder.train()
    for epoch in range(num_epoch):
        train_iter.init_epoch()
        loss_avg = torch.Tensor([0])
        loss_avg = to_cuda(loss_avg, ifcuda)
        for batch in train_iter:
            batch.src = to_cuda(batch.src, ifcuda)
            batch.trg = to_cuda(batch.trg, ifcuda)
            # zero gradients
            encoder.zero_grad()
            decoder.zero_grad()
            
            # pass src sentence to encoder
            output, hidden = encoder(batch.src)
            
            # decoder
            output = decoder(batch.trg, hidden)
            
            # calculate loss
            output = output.transpose(0,2).transpose(0,1)
            target = batch.trg[1:].t()
            loss = lossfn(output, target)
            loss_avg += loss.data
            
            # back prop
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # print information
            if train_iter.iterations % 500 is 0:
                print('Epoch %d/%d, iteration %d --- average loss: %f (time elapsed %s)'
                      % (epoch+1, num_epoch, train_iter.iterations,
                         loss_avg / train_iter.iterations, timeSince(start)))
        
        print('Ending of epoch %d/%d --- average loss: %f (time elapsed %s)'
              % (epoch+1, num_epoch, loss_avg / train_iter.iterations, timeSince(start)))
    
    print('Finished training!\n')
    
    return encoder, decoder


def validating(val_iter, encoder, decoder, lossfn, ifcuda = False):
    encoder = to_cuda(encoder, ifcuda)
    decoder = to_cuda(decoder, ifcuda)
    encoder.eval()
    decoder.eval()
    loss_avg = torch.Tensor([0])
    loss_avg = to_cuda(loss_avg, ifcuda)
    num_example = 0
    for batch in val_iter:
        batch.src = to_cuda(batch.src, ifcuda)
        batch.trg = to_cuda(batch.trg, ifcuda)
        batch.src.volatile = True
        batch.trg.volatile = True
        output, hidden = encoder(batch.src)
        output = decoder(batch.trg, hidden)
        output = output.transpose(0,2).transpose(0,1)
        target = batch.trg[1:].t()
        loss = lossfn(output, target)
        loss_avg += loss.data
        num_example += batch.batch_size * batch.trg.size(0)    # average onto each word
        
    loss_avg /= num_example
    print('Average loss on validation set: %f\n' % (loss_avg))
    
    return loss_avg


def trainingAttn(train_iter, encoder, decoder, lossfn, lr, num_epoch, ifcuda = False):
    encoder = to_cuda(encoder, ifcuda)
    decoder = to_cuda(decoder, ifcuda)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr = lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = lr)
    start = time.time()
    encoder.train()
    decoder.train()
    for epoch in range(num_epoch):
        train_iter.init_epoch()
        loss_avg = torch.Tensor([0])
        loss_avg = to_cuda(loss_avg, ifcuda)
        for batch in train_iter:
            batch.src = to_cuda(batch.src, ifcuda)
            batch.trg = to_cuda(batch.trg, ifcuda)
            # zero gradients
            encoder.zero_grad()
            decoder.zero_grad()
            
            # pass src sentence to encoder
            encoder_outputs, hidden = encoder(batch.src)
            
            # decoder
            hidden_init = hidden[0]           # this is only needed for LSTM encoder (not GRU)
            output = decoder(batch.trg, hidden_init, encoder_outputs, ifcuda)
            
            # calculate loss
            output = output.transpose(0,2).transpose(0,1)
            target = batch.trg[1:].t()
            loss = lossfn(output, target)
            loss_avg += loss.data
            
            # back prop
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # print information
            if train_iter.iterations % 100 is 0:
                print('Epoch %d/%d, iteration %d --- average loss: %f (time elapsed %s)'
                      % (epoch+1, num_epoch, train_iter.iterations,
                         loss_avg / train_iter.iterations, timeSince(start)))
        
        print('Ending of epoch %d/%d --- average loss: %f (time elapsed %s)'
              % (epoch+1, num_epoch, loss_avg / train_iter.iterations, timeSince(start)))
    
    print('Finished training!\n')
    
    return encoder, decoder


def validatingAttn(val_iter, encoder, decoder, lossfn, ifcuda = False):
    encoder = to_cuda(encoder, ifcuda)
    decoder = to_cuda(decoder, ifcuda)
    encoder.eval()
    decoder.eval()
    loss_avg = torch.Tensor([0])
    loss_avg = to_cuda(loss_avg, ifcuda)
    num_example = 0
    for batch in val_iter:
        batch.src = to_cuda(batch.src, ifcuda)
        batch.trg = to_cuda(batch.trg, ifcuda)
        batch.src.volatile = True
        batch.trg.volatile = True
        encoder_outputs, hidden = encoder(batch.src)
        hidden_init = hidden[0]                   # this is needed for LSTM encoder
        output = decoder(batch.trg, hidden_init, encoder_outputs, ifcuda)
        output = output.transpose(0,2).transpose(0,1)
        target = batch.trg[1:].t()
        loss = lossfn(output, target)
        loss_avg += loss.data
        num_example += batch.batch_size * batch.trg.size(0)    # average onto each word
        
    loss_avg /= num_example
    print('Average loss on validation set: %f\n' % (loss_avg))
    
    return loss_avg
