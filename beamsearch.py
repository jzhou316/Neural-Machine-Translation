# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from utils import to_cuda


class BeamUnit():
    def __init__(self, self_id, pre_loc, score, hidden_next, attn_dist = None):
        self.score = score
        self.self_id = self_id
        self.pre_loc = pre_loc
        self.hidden_next = hidden_next
        self.attn_dist = attn_dist


class Beam():
    def __init__(self, K, pad_id, bos_id, eos_id, hidden_init):
        self.K = K
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.step = 0
        self.Kscores = torch.Tensor([0] * K)
        self.beamseq = [[BeamUnit(self_id = bos_id, pre_loc = None, score = 0,
                                  hidden_next = hidden_init)]
                        for k in range(K)]
        
    def beamstep(self, decoder, ifcuda = False, ifAttn = False, *args, **kwargs):      # one step of beam search
        tmp1 = to_cuda(torch.Tensor(1), ifcuda)
        tmp2 = to_cuda(torch.LongTensor(1), ifcuda)
        tmp3 = torch.LongTensor(1)
        m = []
        att = []
        hid = []
        for k in range(self.K):
            if self.beamseq[k][-1].self_id is not self.eos_id:
                m.append(k)
                word_id = Variable(torch.LongTensor([self.beamseq[k][-1].self_id]))
                word_id = to_cuda(word_id, ifcuda)
                hidden = self.beamseq[k][-1].hidden_next
                if ifAttn:
                    output, hidden, attn_dist = decoder.predict(word_id, hidden = hidden, *args, **kwargs)
                    att.append(attn_dist)
                else:
                    output, hidden = decoder.predict(word_id, hidden = hidden, *args, **kwargs)
                    att.append([])
                hid.append(hidden)
                
                scores, inds = output.data.topk(self.K)
            
                tmp1 = torch.cat([tmp1, scores + self.beamseq[k][-1].score])
                tmp2 = torch.cat([tmp2, inds])
                tmp3 = torch.cat([tmp3, torch.LongTensor([k] * self.K)])
        
        if len(m) == 0:
            print('All beams have met <EOS>!')
            return
        
        tmp1 = tmp1[1:]
        tmp2 = tmp2[1:]
        tmp3 = tmp3[1:]
        
        order = tmp1.topk(len(m))[1]
        
        for k, d in enumerate(m):
            score = tmp1[order[k]]
            self_id = tmp2[order[k]]
            pre_loc = tmp3[order[k]]
            if ifAttn:
                self.beamseq[d].append(BeamUnit(self_id, pre_loc, score,
                                       hid[tmp3[order[k]]], att[tmp3[order[k]]]))
            else:
                self.beamseq[d].append(BeamUnit(self_id, pre_loc, score,
                                       hid[tmp3[order[k]]]))
            self.Kscores[d] = score
        
        self.step += 1
        return
    
    def retrieve(self):                     # retrieve the best scored sentence
        k = self.Kscores.topk(1)[1]
        k = k[0]                            # from torch.LongTensor to int
        seq_id = [self.beamseq[k][-1].self_id]
        pre_loc = self.beamseq[k][-1].pre_loc
        m = len(self.beamseq[k]) - 1
        attn = [self.beamseq[k][-1].attn_dist]
        while pre_loc is not None:
            m -= 1
            seq_id.append(self.beamseq[pre_loc][m].self_id)
            attn.append(self.beamseq[pre_loc][m].attn_dist)
            pre_loc = self.beamseq[pre_loc][m].pre_loc
        seq_id = seq_id[::-1]             # reverse the list
        attn = attn[::-1]
            
        return seq_id, attn
