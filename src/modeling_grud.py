# References: 
# https://arxiv.org/pdf/1606.01865.pdf
# https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
# https://github.com/PeterChe1990/GRU-D/blob/master/nn_utils/grud_layers.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from typing import List, Optional
from models import TimeSeriesModel



class GRUDCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dropout):
        super(GRUDCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_gamma = nn.Parameter(torch.zeros(1,input_size), requires_grad=True)
        self.b_gamma = nn.Parameter(torch.zeros(1,input_size), requires_grad=True)
        self.W_gamma_h = nn.Parameter(torch.zeros(input_size, hidden_size), requires_grad=True)
        self.b_gamma_h = nn.Parameter(torch.zeros(1,hidden_size), requires_grad=True)

        self.W_z = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.W_r = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.W = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.U_z = nn.Parameter(torch.empty((hidden_size, hidden_size)), requires_grad=True)
        self.U_r = nn.Parameter(torch.empty((hidden_size, hidden_size)), requires_grad=True)
        self.U = nn.Parameter(torch.empty((hidden_size, hidden_size)), requires_grad=True)
        self.V_z = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.V_r = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.V = nn.Parameter(torch.empty((input_size, hidden_size)), requires_grad=True)
        self.b_z = nn.Parameter(torch.zeros(1,hidden_size), requires_grad=True)
        self.b_r = nn.Parameter(torch.zeros(1,hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1,hidden_size), requires_grad=True)

        nn.init.xavier_uniform_(self.W_z)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V_z)
        nn.init.xavier_uniform_(self.V_r)
        nn.init.xavier_uniform_(self.V)
        nn.init.orthogonal_(self.U_z)
        nn.init.orthogonal_(self.U_r)
        nn.init.orthogonal_(self.U)

        self.reset_dropout_masks()

    def reset_dropout_masks(self):
        self._dropout_mask = [torch.tensor(0),torch.tensor(0),torch.tensor(0)]
        self._recurrent_dropout_mask = [torch.tensor(0),torch.tensor(0),torch.tensor(0)]
        self._masking_dropout_mask = [torch.tensor(0),torch.tensor(0),torch.tensor(0)]

    @jit.script_method
    def forward(self, input, state):
        # type: (Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        x_t, m_t, delta_t = input # (bsz,V) (bsz,V) (bsz)
        h_tm1, xprev = state # (bsz,V) (bsz,V) (bsz,d)
        xprev = xprev*(1-m_t) + m_t*x_t

        # create dropout masks
        if self.dropout>0 and self._dropout_mask[0].ndim==0:
            self._dropout_mask = [F.dropout(torch.ones_like(x_t), 
                                  self.dropout, self.training) for i in range(3)]
            self._recurrent_dropout_mask = [F.dropout(torch.ones_like(h_tm1),
                                             self.dropout, self.training) for i in range(3)]
            self._masking_dropout_mask  = [F.dropout(torch.ones_like(m_t), 
                                            self.dropout, self.training) for i in range(3)]
            
        gamma_t = torch.exp(-torch.clip(self.W_gamma*delta_t+self.b_gamma,min=0))
        x_t = m_t*x_t + (1-m_t)*gamma_t*xprev
        gamma_ht = torch.exp(-torch.clip(torch.matmul(delta_t,self.W_gamma_h)+self.b_gamma_h,min=0))
        h_tm1 = gamma_ht*h_tm1

        # apply 3 dropout masks to x_t, m_t, h_tm1
        if self.dropout>0:
            x_t_z = self._dropout_mask[0]*x_t
            x_t_r = self._dropout_mask[1]*x_t
            x_t_h = self._dropout_mask[2]*x_t
            m_t_z = self._masking_dropout_mask[0]*m_t
            m_t_r = self._masking_dropout_mask[1]*m_t
            m_t_h = self._masking_dropout_mask[2]*m_t
            h_tm1_z = self._recurrent_dropout_mask[0]*h_tm1
            h_tm1_r = self._recurrent_dropout_mask[1]*h_tm1
            h_tm1_h = self._recurrent_dropout_mask[2]*h_tm1
        else:
            x_t_z = x_t_r = x_t_h = x_t
            m_t_z = m_t_r = m_t_h = m_t
            h_tm1_z = h_tm1_r = h_tm1_h = h_tm1

        z_t = torch.sigmoid(torch.matmul(x_t_z, self.W_z)
                            +torch.matmul(h_tm1_z, self.U_z)
                            +torch.matmul(m_t_z,self.V_z)+self.b_z)
        r_t = torch.sigmoid(torch.matmul(x_t_r, self.W_r)
                            +torch.matmul(h_tm1_r, self.U_r)
                            +torch.matmul(m_t_r,self.V_r)+self.b_r)
        h_t = torch.tanh(torch.matmul(x_t_h,self.W)
                         +torch.matmul(r_t*h_tm1_h, self.U)
                         +torch.matmul(m_t_h,self.V)+self.b)
        h_t = (1-z_t)*h_tm1 + z_t*h_t

        return h_t, (h_t, xprev)
    

class GRUD(jit.ScriptModule):
    def __init__(self, *cell_args):
        super(GRUD, self).__init__()
        self.cell = GRUDCell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tensor
        x_t = input[0].unbind(1) # T: bsz, V
        m_t = input[1].unbind(1) # T: bsz, V
        delta_t = input[2].unbind(1) # T: bsz, V
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        self.cell.reset_dropout_masks()
        for i in range(len(x_t)):
            out, state = self.cell((x_t[i],m_t[i],delta_t[i]), state) # (bsz,h) ((bsz,hid), (bsz,V))
            outputs += [out]
        return torch.stack(outputs).transpose(0,1)
    


class GRUD_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)
        self.grud = GRUD(args.V, args.hid_dim, args.dropout)
        self.V = args.V
        self.hid_dim = args.hid_dim
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x_t, m_t, delta_t, seq_len, demo, labels=None):
        bsz = x_t.size()[0]
        device = x_t.device
        initial_state = (torch.zeros((bsz,self.hid_dim), device=device), 
                         torch.zeros((bsz,self.V), device=device)) 
        ts_emb = self.grud((x_t,m_t,delta_t),initial_state) # bsz, T, d
        bsz, max_len, d = ts_emb.size()
        index = (seq_len-1)[:,None,None].repeat((1,1,d))
        ts_emb = torch.gather(ts_emb, 1, index)[:,0,:]
        ts_emb = self.dropout(ts_emb)

        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        # prediction
        logits = self.binary_head(ts_demo_emb)[:,0]
        # prediction/loss
        return self.binary_cls_final(logits, labels)
    


    

if __name__=='__main__':
    input_size, hidden_size, dropout = 10, 64, 0.2
    V = input_size
    bsz = 4
    seq_len = 126
    model = GRUD(input_size, hidden_size, dropout).cuda()
    input = [torch.zeros((bsz,seq_len,V)).cuda(), torch.zeros((bsz,seq_len,V)).cuda(), 
             torch.zeros((bsz,seq_len,V)).cuda()]
    state = [torch.zeros((bsz,hidden_size)).cuda(),torch.zeros((bsz,V)).cuda()]
    emb = model (input,state) 
    print (emb.size())