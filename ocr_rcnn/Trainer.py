import torch
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from Common import AccuracyMeasure, Eval, LabelCodec
from tqdm import *


class Trainer:
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.data_train = opt['data_train']
        self.data_val = opt['data_val']
        self.model = opt['model']
        self.criterion = opt['criterion']
        self.optimizer = opt['optimizer']
        self.schedule = opt['schedule']
        self.converter = LabelCodec(opt['alphabet'])
        self.evaluator = Eval()
        print('Scheduling is {}'.format(self.schedule))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=opt['epochs'])
        self.batch_size = opt['batch_size']
        self.count = opt['epoch']
        self.epochs = opt['epochs']
        self.cuda = opt['cuda']
        self.collate_fn = opt['collate_fn']
        self.init_meters()

    def init_meters(self):
        self.avgTrainLoss = AccuracyMeasure("Train loss")
        self.avgTrainCharAccuracy = AccuracyMeasure("Train Character Accuracy")
        self.avgTrainWordAccuracy = AccuracyMeasure("Train Word Accuracy")
        self.avgValLoss = AccuracyMeasure("Validation loss")
        self.avgValCharAccuracy = AccuracyMeasure("Validation Character Accuracy")
        self.avgValWordAccuracy = AccuracyMeasure("Validation Word Accuracy")

    def forward(self, x):
        logits = self.model(x)
        return logits.transpose(1, 0)

    def loss_fn(self, logits, targets, pred_sizes, target_sizes):
        loss = self.criterion(logits, targets, pred_sizes, target_sizes)
        return loss

    def step(self):
        self.max_grad_norm = 0.05
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
    
    def schedule_lr(self):
        if self.schedule:
            self.scheduler.step()

    def run_batch(self, batch, report_accuracy=False, validation=False):
        input_, targets = batch['img'], batch['label']
        targets, lengths = self.converter.encode(targets)
        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.log_softmax(logits, 2)
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        targets= targets.view(-1).contiguous()
        loss = self.loss_fn(logits, targets, pred_sizes, lengths)
        if report_accuracy:
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, pred_sizes.data, raw=False)
            ca = np.mean((list(map(self.evaluator.char_accuracy, list(zip(sim_preds, batch['label']))))))
            wa = np.mean((list(map(self.evaluator.word_accuracy, list(zip(sim_preds, batch['label']))))))
        return loss, ca, wa

    def run_epoch(self, validation=False):
        if not validation:
            loader = torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)
            print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
            print(f"Epoch: [{self.count}]\[{self.epochs}]")
            pbar = tqdm(loader, desc='Training', leave=True)
            self.model.train()
        else:
            loader = torch.utils.data.DataLoader(self.data_val, batch_size=self.batch_size, collate_fn=self.collate_fn)
            pbar = tqdm(loader, desc='Validating', leave=True)
            self.model.eval()
        outputs = []
        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.training_step(batch)
            else:
                output = self.validation_step(batch)
            pbar.set_postfix(output)
            outputs.append(output)
        self.schedule_lr()
        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.validation_end(outputs)
        return result

    def training_step(self, batch):
        loss, ca, wa = self.run_batch(batch, report_accuracy=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        result = OrderedDict({
                'TrainLoss': abs(loss.item()),
                'CharAccuracy': ca.item(),
                'WordAccuracy': wa.item()
            })
        return result

    def validation_step(self, batch):
        loss, ca, wa = self.run_batch(batch, report_accuracy=True, validation=True)
        result = OrderedDict({
                'ValLoss': abs(loss.item()),
                'CharAccuracy': ca.item(),
                'WordAccuracy': wa.item()
            })
        return result

    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['TrainLoss'])
            self.avgTrainCharAccuracy.add(output['CharAccuracy'])
            self.avgTrainWordAccuracy.add(output['WordAccuracy'])

        train_loss_mean = abs(self.avgTrainLoss.compute())
        train_ca_mean = self.avgTrainCharAccuracy.compute()
        train_wa_mean = self.avgTrainWordAccuracy.compute()
        result = {
            'train_loss': train_loss_mean, 
            'train_ca': train_ca_mean,
            'train_wa': train_wa_mean
            }
        return result

    def validation_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['ValLoss'])
            self.avgValCharAccuracy.add(output['CharAccuracy'])
            self.avgValWordAccuracy.add(output['WordAccuracy'])

        val_loss_mean = abs(self.avgValLoss.compute())
        val_ca_mean = self.avgValCharAccuracy.compute()
        val_wa_mean = self.avgValWordAccuracy.compute()
        result = {
            'val_loss': val_loss_mean, 
            'val_ca': val_ca_mean,
            'val_wa': val_wa_mean
            }
        return result