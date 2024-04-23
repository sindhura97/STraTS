# References:
# https://github.com/mlds-lab/interp-net/blob/master/src/multivariate_example.py#L23
# https://openreview.net/pdf?id=r1efr3C9Ym
from models import TimeSeriesModel
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class SingleChannelInterp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(1,1,1,args.V),
                                   requires_grad=True)
        self.hours_look_ahead = args.hours_look_ahead
        self.ref_points = args.ref_points
        self.ref_t = nn.Parameter(torch.linspace(0, 
                            self.hours_look_ahead, self.ref_points),
                            requires_grad=False)
    
    def forward(self, x,m,t,h, reconstruction=False):
        if reconstruction:
            m = h
            ref_t = t
        else:
            ref_t = self.ref_t[None,:]
        # x,m: bsz, T, V
        # t: bsz, T
        # ref_t: bsz(1), T'
        weights = (t[:,:,None]-ref_t[:,None,:])**2 # bsz,T,T'
        pos_kernel = torch.log(1+torch.exp(self.kernel)) # 1,1,1,V
        weights = pos_kernel*weights[:,:,:,None] # bsz,T,T',V
        weights_lp = torch.exp(-weights) # eq (1)
        weights_lp = weights_lp*m[:,:,None,:] # bsz,T,T',V
        lambda_ = weights_lp.sum(dim=1) # bsz,T',V
        sigma = (weights_lp*x[:,:,None,:]).sum(dim=1) # bsz,T',V
        sigma = sigma/torch.clip(lambda_,min=1)
        if reconstruction:
            return sigma, lambda_
        weights_hp = torch.exp(-10.0*weights)
        weights_hp = weights_hp*m[:,:,None,:] # bsz,T,T',V
        lambda_hp = weights_hp.sum(dim=1) # bsz,T',V
        gamma = (weights_hp*x[:,:,None,:]).sum(dim=1) # bsz,T',V
        gamma = gamma/torch.clip(lambda_hp,min=1)
        return sigma, lambda_, gamma

        
class CrossChannelInterp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rho = nn.Parameter(torch.eye((args.V))[None,None,:,:], 
                                requires_grad=True)

    def forward(self, sigma, lambda_, gamma=None):
        sigma = sigma[:,:,:,None] # bsz,T',V,1
        lambda_ = lambda_[:,:,:,None] # bsz,T',V,1
        chi = (self.rho*lambda_*sigma).sum(dim=2) # bsz,T',V
        chi = chi / torch.clip(lambda_.sum(dim=2),min=1) # bsz,T',V
        if gamma is None:
            return chi
        tau = gamma-chi
        return chi, tau
    




class InterpNet(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)
        self.sci = SingleChannelInterp(args)
        self.cci = CrossChannelInterp(args)
        self.gru = nn.GRU(args.V*3, args.hid_dim, batch_first=True,
                          dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)

    def custom_loss(self, x, m, h, aux_output):
        loss = (x-aux_output)**2
        loss_mask = m*(1-h)
        loss = loss*loss_mask
        loss = loss.mean(dim=1)/torch.clip(loss_mask.sum(dim=1),min=1) # bsz, V
        loss = loss.sum(dim=1)/x.size()[-1]
        return loss.mean()
        
    def forward(self, x,m,t,h, demo, labels=None):
        sigma, lambda_, gamma = self.sci(x,m,t,h)
        chi, tau = self.cci(sigma, lambda_, gamma)
        ts = torch.cat((lambda_,chi,tau),dim=-1) # bsz,T',3V
        ts_emb = self.gru(ts)[1].reshape((ts.size()[0],-1)) # bsz,d

        ts_emb = self.dropout(ts_emb)
        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        # prediction
        logits = self.binary_head(ts_demo_emb)[:,0]
        # prediction/loss
        if labels is None:
            return F.sigmoid(logits)
        main_loss = F.binary_cross_entropy_with_logits(logits, labels, 
                                    pos_weight=self.pos_class_weight)
        

        sigma, lambda_ = self.sci(x,m,t,h,reconstruction=True)
        aux_output = self.cci(sigma, lambda_) # bsz,T,V
        aux_loss = self.custom_loss(x,m,h,aux_output)
        return main_loss+aux_loss