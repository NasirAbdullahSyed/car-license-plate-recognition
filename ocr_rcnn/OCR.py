import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import random_split
from Common import TrainingMonitor, wrap_dir
from Trainer import Trainer
from DatasetLoader import LoadDataset, CollateDataset
from CRNN import CRNN
from CTCLoss import CTCLossWrapper
from tqdm import *
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device:', device)

class OCR:
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.savepath = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available() 
        self.cuda_count = torch.cuda.device_count()
        if self.cuda:
            print("Using GPU(s) for model training.")
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)
        self.epoch = 0
        self.best_score = None
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint['epoch']
            self.best_score = self.checkpoint['best']
            self.load()
        else:
            print('No existing model found!')

    def fit(self, param):
        param['cuda'] = self.cuda
        param['model'] = self.model
        param['optimizer'] = self.optimizer
        # Init a log file
        logging.basicConfig(filename=f"{param['log_dir']}/{param['name']}.csv", level=logging.INFO)
        # Init the training monitor
        self.saver = TrainingMonitor(self.savepath, stop_count=10, verbose=True, best_score=self.best_score)
        param['epoch'] = self.epoch
        trainer = Trainer(param)
        for epoch in range(param['epoch'], param['epochs']):
            trainer.count = epoch + 1
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            if epoch == 0:
                logging.info("Epoch, Training Loss, Validation Loss, Train Char Accuracy, Validation Char Accuracy, Train Word Accuracy, Validation Word Accuracy")
            logging.info(f"{epoch}, {train_result['train_loss']:.6f}, {val_result['val_loss']:.6f}, \
                            {train_result['train_ca']:.6f}, {val_result['val_ca']:.6f}, \
                            {train_result['train_wa']:.6f}, {val_result['val_wa']:.6f}")
            self.val_loss = val_result['val_loss']
            print(f"Resultant Validation Loss: {self.val_loss}")
            if self.savepath:
                self.saver(self.val_loss, epoch, self.model, self.optimizer)
            if self.saver.stop_training:
                print("Training Monitor halted the training process as validation loss failed decrease in successive epochs.")
                break

    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(self.savepath, self.checkpoint['epoch']))
        state_dict = self.checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])


classes = """;Lv4YT"2iNP)MrJj QUh8+RgmaoDI?$ncxtA-W#V/@K!6,:OXFubl0yqwzk_93pf'd*sEBGH17e5S.C(%"""
params = {
    'name': 'test1',
    'local_path': 'datasets',
    'img_dir': 'ocr_train',
    'imgH': 32,
    'n_classes': len(classes),
    'lr': 0.001,
    'epochs': 16,
    'batch_size': 32,
    'save_dir': 'trained_models',
    'log_dir': 'logs',
    'resume': False,
    'cuda': False,
    'schedule': False 
}

data = LoadDataset(params)
params['collate_fn'] = CollateDataset()
train_split = int(0.8*len(data))
val_split = len(data) - train_split
params['data_train'], params['data_val'] = random_split(data, (train_split, val_split))
print(f"Traininig Data Size: {len(params['data_train'])}")
print(f"Validation Data Size: {len(params['data_val'])}")
params['alphabet'] = classes
model = CRNN(params)
params['criterion'] = CTCLossWrapper()
save_path = os.path.join(params['save_dir'], params['name'])
wrap_dir(save_path)
wrap_dir(params['log_dir'])
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
ocr = OCR(model, optimizer, savepath=save_path, resume=params['resume'])
ocr.fit(params)