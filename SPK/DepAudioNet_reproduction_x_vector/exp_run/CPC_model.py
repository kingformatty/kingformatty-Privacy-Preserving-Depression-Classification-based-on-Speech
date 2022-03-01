from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb
class CDCK2(nn.Module):
    def __init__(self):

        super(CDCK2, self).__init__()

        self.encoder = nn.Sequential( # downsampling factor = 160
            #nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            #nn.Conv1d(40,128, kernel_size = 10, stride = 5, padding=3, bias = False),
            #nn.BatchNorm1d(128),#512
            #nn.ReLU(inplace=True),
            #nn.Conv1d(40, 128, kernel_size=8, stride=4, padding=2, bias=False),
            #nn.BatchNorm1d(128),
            #nn.ReLU(inplace=True),
            nn.Conv1d(40, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
            #nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm1d(128),
            #nn.ReLU(inplace=True),
            #nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm1d(128),
            #nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(128, 128, num_layers=2, bidirectional=False, batch_first=True)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.fc = FullyConnected(in_channels = 128,
                                 out_channels = 1,
                                 activation='sigmoid',
                                 normalisation=None)

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, 128).cuda()
        else: return torch.zeros(1, batch_size, 128)

    def forward(self, x):
        hidden = torch.Tensor(128)
        batch = x.size()[0]
        #t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long() # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        output, hidden = self.gru(z) # output size e.g. 8*100*256
        #c_t = output[:,t_samples,:].view(batch, 128) # c_t e.g. size 8*256
        #pred = torch.empty((self.timestep,batch,128)).float() # e.g. size 12*8*512
        #for i in np.arange(0, self.timestep):
        #    linear = self.Wk[i]
        #    pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        #for i in np.arange(0, self.timestep):
        #    total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
        #    correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
        #    nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        #nce /= -1.*batch*self.timestep
        #accuracy = 1.*correct.item()/batch
        output = output[:,-1,:]
        x = self.fc(output)
        return x

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) # output size e.g. 8*128*256
        
        return output, hidden # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')


    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)        

        return x

