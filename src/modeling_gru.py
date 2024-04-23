from argparse import Namespace
from models import TimeSeriesModel
import torch.nn as nn
import torch



class GRU_TS(TimeSeriesModel):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.gru = nn.GRU(args.V*3, args.hid_dim, batch_first=True,
                          dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, ts, demo, labels=None):
        ts_emb = self.gru(ts)[1].reshape((ts.size()[0],-1))
        ts_emb = self.dropout(ts_emb)
        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        # prediction
        logits = self.binary_head(ts_demo_emb)[:,0]
        # prediction/loss
        return self.binary_cls_final(logits, labels)
    