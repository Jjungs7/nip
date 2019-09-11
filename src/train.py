import numpy as np
from NSC import NSC
from dataset import NIPDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import sys


class Instructor:
    def __init__(self, args):
        self.args = args
        self.model = NSC(self.args)
        if args.resume:
            self.model_load(args.resume)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()

    def model_save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self.model.state_dict(), fname)

    def model_load(self, fname):
        self.model.load_state_dict(torch.load(fname))
        return self.model

    def train(self):
        recent_loss = sys.float_info.max
        counter = 1
        self.model.train()
        train_dataset = NIPDataset(args=self.args, data_path=self.args.data_path_prefix + "train-dummy.txt")
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                      collate_fn=train_dataset.custom_collate_fn)
        for i_epoch in range(1, self.args.epochs):
            for i_sample, sample_batch in enumerate(train_dataloader):
                counter += 1
                (text, length, mask, label) = sample_batch
                if self.args.device is 'cuda':
                    text, length, mask, label = text.cuda(), length.cuda(), mask.cuda(), label.cuda()

                pred = self.model(text, length, mask)  # N, 3
                loss = self.criterion(pred, label)
                recent_loss = min(recent_loss, loss.item())
                current_loss = loss.item()
                loss.backward()
                self.optimizer.update()
                self.model.zero_grad()
                self.optimizer.zero_grad()

                if counter % self.args.log_interval == 0:
                    val_losses = self.validation()
                    self.model.train()
                    print("Epoch: {}/{}...".format(i_epoch + 1, self.args.max_epochs),
                          "Counter: {}...".format(counter),
                          "Loss: {:.6f}...".format(current_loss),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

                    if current_loss <= recent_loss:
                        self.model_save(os.path.join(self.args.model_save_path, 'model_epoch{}_counter{}.pth'.format(i_epoch, counter)))


    def validation(self):
        val_losses = []
        dev_dataset = NIPDataset(args=self.args, data_path=self.args.data_path_prefix + "dev-dummy.txt")
        dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=self.args.batch_size, shuffle=False,
                                         collate_fn=dev_dataset.custom_collate_fn)
        for i_sample, sample_batch in enumerate(dev_dataloader):
            (text, length, mask, label) = sample_batch
            self.model.eval()

            if self.args.device is 'cuda':
                text, length, mask, label = text.cuda(), length.cuda(), mask.cuda(), label.cuda()

            pred = self.model(text, length, mask)  # N, 3
            val_loss = self.criterion(pred, label)
            val_losses.append(val_loss.item())

        return val_losses