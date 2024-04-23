# References:
# https://arxiv.org/pdf/1711.03905.pdf
# https://github.com/khirotaka/SAnD/tree/master


from argparse import Namespace
from models import TimeSeriesModel
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.hid_dim%args.num_heads==0
        self.dk = args.hid_dim//args.num_heads
        self.Wq = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)),
                               requires_grad=True)
        self.Wk = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)),
                               requires_grad=True)
        self.Wv = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)),
                               requires_grad=True)
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        self.Wo = nn.Linear(args.hid_dim,args.hid_dim,bias=False)
        self.num_heads = args.num_heads
        self.dropout = args.dropout

    def forward(self, x, mask):
        # x: bsz, T, d
        bsz, T, d = x.size()
        device = x.device
        queries = torch.matmul(x, self.Wq).view(bsz, T, self.num_heads, self.dk)/np.sqrt(self.dk)
        keys = torch.matmul(x, self.Wk).view(bsz, T, self.num_heads, self.dk)
        values = torch.matmul(x, self.Wv).view(bsz, T, self.num_heads, self.dk)
        A = torch.einsum('bthd,blhd->bhtl', queries, keys)+mask # bsz, h, T, T
        A = F.softmax(A, dim=-1)
        A = F.dropout(A, self.dropout)
        x = torch.einsum('bhtl,bthd->bhtd', A, values)
        x = self.Wo(x.reshape((bsz,T,d)))
        return x


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(args.hid_dim, args.hid_dim* 2, 1),
            nn.ReLU(),
            nn.Conv1d(args.hid_dim*2, args.hid_dim, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(args)
        self.norm_mha = nn.LayerNorm(args.hid_dim)
        self.norm_ffn = nn.LayerNorm(args.hid_dim)
        self.dropout = args.dropout

    def forward(self, x, mask):
        x2 = F.dropout(self.mha(x, mask), self.dropout, self.training)
        x = self.norm_mha(x+x2)
        x2 = F.dropout(self.ffn(x), self.dropout, self.training)
        x = self.norm_ffn(x+x2)
        return x
    

class DenseInterpolation(nn.Module):
    def __init__(self, args):
        super().__init__()
        cols = torch.arange(args.M).reshape((1,args.M))/args.M
        rows = torch.arange(args.T).reshape((args.T,1))/args.T
        self.W = (1-torch.abs(rows-cols))**2
        self.W = nn.Parameter(self.W, requires_grad=False)

    def forward(self, x):
        bsz = x.size()[0]
        x = torch.matmul(x.transpose(1,2),self.W) # bsz,V,M
        return x.reshape((bsz,-1))


class SAND(TimeSeriesModel):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.input_embedding = nn.Conv1d(args.V*3, args.hid_dim, 1)
        self.positional_encoding = nn.Parameter(torch.empty((1,args.T,args.hid_dim)),
                                                requires_grad=True)
        nn.init.normal_(self.positional_encoding)
        indices = torch.arange(args.T)
        # t attends to t-r,...,t
        self.mask = torch.logical_and(indices[None,:]<=indices[:,None], 
                                      indices[None,:]>=indices[:,None]-args.r).float()
        self.mask = (1-self.mask)*torch.finfo(self.mask.dtype).min
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.dropout = args.dropout
        self.transformer = nn.ModuleList([TransformerBlock(args) 
                                          for i in range(args.num_layers)])
        self.dense_interpolation = DenseInterpolation(args)

    def forward(self, ts, demo, labels=None):
        ts_inp_emb = self.input_embedding(ts.permute((0,2,1))).permute((0,2,1))
        ts_inp_emb = ts_inp_emb + self.positional_encoding
        if self.dropout>0:
            ts_inp_emb = F.dropout(ts_inp_emb, self.dropout, self.training)
        ts_hid_emb = ts_inp_emb
        for layer in self.transformer:
            ts_hid_emb = layer(ts_hid_emb, self.mask)
        ts_emb = self.dense_interpolation(ts_hid_emb)
        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        # prediction
        logits = self.binary_head(ts_demo_emb)[:,0]
        # prediction/loss
        return self.binary_cls_final(logits, labels)
    