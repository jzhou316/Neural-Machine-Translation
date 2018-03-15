from utils import loadIWSLT

batch_size = 64
train_iter, val_iter, DE, EN, train, val, test = loadIWSLT(batch_size)


def itosyms(seq, dic):
    return [dic[seq[s].data[0]] for s in range(len(seq))]

val_iter.init_epoch()
batch = next(iter(val_iter))
f = open('val1stbatch.txt', 'w')
for k in range(batch.batch_size):
    print(' '.join(itosyms(batch.src[:,k], DE.vocab.itos)), file = f)
    print('--->', file = f)
    print(' '.join(itosyms(batch.trg[:,k], EN.vocab.itos)), file = f)
    print('\n', file = f)

f.close()
