import numpy as np
from nsc import NSC
from dataset import NIPDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import sys
import datetime
import math


class Instructor:
    def __init__(self, args):
        self.args = args
        self.model = NSC(self.args)
        if args.load_model or args.test:
            self.model = self.model_load(args.load_model)
        if self.args.device is 'cuda':
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        if not args.test:
            self.train_dataset = NIPDataset(args=self.args, data_path=self.args.data_path_prefix + "train.txt")
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                          collate_fn=self.train_dataset.custom_collate_fn)
            self.dev_dataset = NIPDataset(args=self.args, data_path=self.args.data_path_prefix + "dev.txt")
            self.dev_dataloader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.batch_size, shuffle=False,
                                             collate_fn=self.dev_dataset.custom_collate_fn)
        else:
            self.test_dataset = NIPDataset(args=self.args, data_path=self.args.data_path_prefix + "test.txt")
            self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size, shuffle=True,
                                              collate_fn=self.test_dataset.custom_collate_fn)


    def model_save(self, fname):
        with open(fname, 'wb') as f:
            torch.save(self.model.state_dict(), fname)

    def model_load(self, fname):
        self.model.load_state_dict(torch.load(fname))
        return self.model

    def train(self):
        recent_loss = sys.float_info.max
        self.model.train()

        for i_epoch in range(1, self.args.epochs+1):
            counter = 1
            for i_sample, sample_batch in enumerate(self.train_dataloader):
                counter += 1
                (text, length, mask, label) = sample_batch
                if self.args.device is 'cuda':
                    text, length, mask, label = text.cuda(), length.cuda(), mask.cuda(), label.cuda()

                self.optimizer.zero_grad()
                pred = self.model(text, length, mask)  # N, 3
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()

                if counter % self.args.log_interval == 0:
                    val_losses = self.validation()
                    current_loss = np.mean(val_losses)
                    self.model.train()
                    print("Epoch: {}/{}...".format(i_epoch, self.args.epochs),
                          "Counter: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(current_loss))

                    if current_loss <= recent_loss:
                        now = datetime.datetime.today().strftime("%m%d_%H%M")
                        self.model_save(os.path.join(self.args.model_save_path, '{}_epoch{}.pth'.format(now, i_epoch)))
                        recent_loss = min(recent_loss, current_loss)
                        print("Model saved as {}_epoch{}.pth".format(now, i_epoch))

    def validation(self):
        val_losses = []
        for i_sample, sample_batch in enumerate(self.dev_dataloader):
            (text, length, mask, label) = sample_batch
            self.model.eval()

            if self.args.device is 'cuda':
                text, length, mask, label = text.cuda(), length.cuda(), mask.cuda(), label.cuda()

            pred = self.model(text, length, mask)  # N, 3
            val_loss = self.criterion(pred, label)
            val_losses.append(val_loss.item())

        return val_losses

    def test(self):
        self.model.eval()
        total = len(self.test_dataset)
        correct = 0
        diff = 0
        for i_sample, sample_batch in enumerate(self.test_dataloader):
            (text, length, mask, label) = sample_batch

            if self.args.device is 'cuda':
                text, length, mask, label = text.cuda(), length.cuda(), mask.cuda(), label.cuda()

            pred = self.model(text, length, mask)
            correct += (torch.argmax(pred, axis=1) == label).sum().item()
            diff += (abs(torch.argmax(pred, axis=1)-label)**2).sum().item()

        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}")

        rmse = math.sqrt(diff / total)
        print(f"RMSE: {rmse:.4f}")
