from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np


class PretrainEvaluator:
    def __init__(self, args):
        self.args = args
        self.io = {}

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = '+split)

        if split not in self.io:
            batches = []
            eval_ind = dataset.splits[split]
            num_samples = len(eval_ind)
            for start in tqdm(range(0,num_samples,self.args.eval_batch_size),
                              desc='generating io for eval split '+split):
                batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
                for i in range(3):
                    batches.append(dataset.get_batch(batch_ind))
            self.io[split] = batches

        model.eval()
        pbar = tqdm(self.io[split], desc='running forward pass')
        loss, count = 0,0
        for batch in pbar:
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                train_loss = model(**batch)
                num_pred = batch['forecast_mask'].sum()
                loss += train_loss*num_pred
                count += num_pred
        result = {'loss_neg':-(loss/count).item()}
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                              +str(train_step)+': '+str(result))
        return result

