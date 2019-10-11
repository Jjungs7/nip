import numpy as np
from abae import ABAE
from absc import ABSC
from word2vec import word2vec
from dataset import NIPDataset
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import torch
import torch.nn as nn
import os
import sys
import datetime
import math
from tqdm import tqdm
from tensorboardX import SummaryWriter
summary = SummaryWriter()


class Instructor:
    def __init__(self, args):
        self.args = args
        self.w2v = word2vec(self.args.tokenized_path)
        self.w2v.embed(self.args.w2v_path, self.args.emsize)
        self.w2v.aspect(self.args.naspects)

        if self.args.category == 'abae':
            self.model = ABAE(self.w2v.E, self.w2v.T)
            if self.args.model_name is not '' or self.args.test:
                self.model = self.model_load(os.path.join(self.args.abae_path, self.args.model_name))
        elif self.args.category == 'absc':
            pass
        else:
            print("You must select abae or absc")
            exit()

        if self.args.device is 'cuda':
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        if not args.test:
            self.train_dataset = NIPDataset(args=self.args, data_path=os.path.join(self.args.data_path_prefix, "train.txt"))
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                          collate_fn=self.train_dataset.custom_collate_fn)
            self.dev_dataset = NIPDataset(args=self.args, data_path=os.path.join(self.args.data_path_prefix, "dev.txt"))
            self.dev_dataloader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.batch_size, shuffle=False,
                                             collate_fn=self.dev_dataset.custom_collate_fn)
        else:
            self.test_dataset = NIPDataset(args=self.args, data_path=os.path.join(self.args.data_path_prefix, "test.txt"))
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
            train_loss_list = []
            valid_loss_list = []
            for i_sample, sample_batch in enumerate(tqdm(self.train_dataloader)):
                counter += 1
                (text, length, mask, label) = sample_batch
                if self.args.device is 'cuda':
                    text, length, mask, label = text.cuda(), length.cuda(), mask.cuda(), label.cuda()

                self.optimizer.zero_grad()
                pred = self.model(text, length, mask)  # N, 3
                loss = self.criterion(pred, label)
                train_loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()

                if counter % self.args.log_interval == 0:
                    val_losses = self.validation()
                    current_loss = np.mean(val_losses)
                    valid_loss_list.append(current_loss)
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

            summary.add_scalar('loss/train_loss', np.mean(train_loss_list), i_epoch)
            summary.add_scalar('loss/valid_loss', np.mean(valid_loss_list), i_epoch)

        summary.close()


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

    def abae_train(self):
        recent_loss = sys.float_info.max
        self.model.train()

        for i_epoch in range(1, self.args.epochs+1):
            counter = 1
            train_loss_list = []
            valid_loss_list = []
            for i_sample, sample_batch in enumerate(tqdm(self.train_dataloader)):
                counter += 1
                pos = sample_batch[0]
                neg = next(iter(self.train_dataloader))[0]
                r_s, z_s, z_n = self.model(pos, neg)
                J = self.max_margin_loss(r_s, z_s, z_n)
                U = self.orthogonal_regularization(self.model.T.weight)
                loss = J + self.args.reg * self.args.batch_size * U
                train_loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if counter % self.args.log_interval == 0:
                    current_loss = self.abae_validatation()
                    valid_loss_list.append(current_loss)
                    self.model.train()
                    print("Epoch: {}/{}...".format(i_epoch, self.args.epochs),
                          "Counter: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(current_loss))

                    if current_loss <= recent_loss:
                        now = datetime.datetime.today().strftime("%m%d_%H%M")
                        self.model_save(os.path.join(self.args.abae_path, '{}_epoch{}.pth'.format(now, i_epoch)))
                        recent_loss = min(recent_loss, current_loss)
                        print("Model saved as {}_epoch{}.pth".format(now, i_epoch))

            summary.add_scalar('loss/train_loss', np.mean(train_loss_list), i_epoch)
            summary.add_scalar('loss/valid_loss', np.mean(valid_loss_list), i_epoch)

        summary.close()

    def max_margin_loss(self, r_s, z_s, z_n):
        device = r_s.device
        pos = torch.bmm(z_s.unsqueeze(1), r_s.unsqueeze(2)).squeeze(2)
        negs = torch.bmm(z_n.unsqueeze(1), r_s.unsqueeze(2)).squeeze(2)
        J = torch.ones(negs.shape).to(device) - pos.expand(negs.shape) + negs
        return torch.sum(torch.clamp(J, min=0.0))

    def orthogonal_regularization(self, T):
        T_n = normalize(T, dim=1)
        I = torch.eye(T_n.shape[0]).to(T_n.device)
        return torch.norm(T_n.mm(T_n.t()) - I)

    def abae_validatation(self):
        self.model.eval()
        val_losses = []
        for i_sample, sample_batch in enumerate(self.dev_dataloader):
            pos = sample_batch[0]
            neg = next(iter(self.dev_dataloader))[0]

            if self.args.device is 'cuda':
                pog, neg = pos.cuda(), neg.cuda()

            r_s, z_s, z_n = self.model(pos, neg)
            J = self.max_margin_loss(r_s, z_s, z_n).item()
            U = self.orthogonal_regularization(self.model.T.weight).item()
            val_losses.append((J + self.args.reg * self.args.batch_size * U))

        return np.mean(val_losses)

    def sample_aspects(self, projection, i2w, n=8):
        projection = torch.sort(projection, dim=1)
        for j, (projs, index) in enumerate(zip(*projection)):
            index = index[-n:].detach().cpu().numpy()
            words = ', '.join([i2w[i] for i in index])
            print('Aspect %2d: %s' % (j, words))
