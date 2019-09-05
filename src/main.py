import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
import numpy as np
import os
from tqdm import tqdm
import dataload, encode
import sys

vocab_to_int, int_to_vocab = encode.make_dict()
MODEL_PATH = 'model'

def main(args):
    model = Model(args.ntokens, args.emsize, args.nclasses, args.sen_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    if args.resume:
        model = model_load(model, args.resume)

    # Dataset
    train_loader = dataload.get_loader('train.txt')
    dev_loader = dataload.get_loader('dev.txt')
    test_loader = dataload.get_loader('test.txt')

    # Train
    train(model, train_loader, dev_loader, optimizer=optimizer, criterion=criterion, args=args)
    # Eval


def train(model, train_loader, dev_loader, optimizer, criterion, args):
    recent_loss = sys.float_info.max

    counter = 0
    clip = args.clip

    if args.device is 'cuda':
        model.cuda()

    model.train()

    for e in tqdm(args.epochs):
        h = model.init_hidden(args.batch_size)

        for inputs, labels in train_loader:
            counter += 1

            if args.devide is 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            h = tuple([each.data for each in h])

            model.zero_grad()

            inputs = inputs.type(torch.IntTensor)
            output, h = model(inputs, h)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % args.log_interval == 0:
                val_h = model.init_hidden(args.batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in dev_loader:
                    val_h = tuple([each.data for each in val_h])

                    if args.devide is 'cuda':
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.IntTensor)
                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                current_loss = loss.item()

                model.train()
                print("Epoch: {}/{}...".format(e + 1, args.epochs),
                      "Counter: {}...".format(counter),
                      "Loss: {:.6f}...".format(current_loss),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

                if current_loss <= recent_loss:
                    model_save(model, os.path.join(MODEL_PATH, 'model_epoch{}_counter{}.pth'.format(e, counter)))


def evaluate(model, test_loader, optimizer, criterion, args):
    total = 0
    correct = 0

    TP = 0
    FP = 0
    FN = 0

    if args.device is 'cuda':
        model.cuda()

    model.eval()

    h = model.init_hidden(args.batch_size)

    # Evaluate
    for inputs, labels in tqdm(test_loader):
        if args.devide is 'cuda':
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs = inputs.type(torch.IntTensor)
        output, h = model(inputs, h)

        _, predicted = torch.max(output, 1)
        total += labels.size(0)

        for i in range(labels.size(0)):
            if predicted[i] == 1:
                if labels[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if labels[i] == 1:
                    FN += 1

            if predicted[i] == labels[i]:
                correct += 1

    print(f"Accuracy: {100*correct/total}")

    recall = TP / (TP+FN)
    precision = TP / (TP+FP)

    print(f"f1 score: {2*(recall * precision)/(recall + precision)}")



def model_save(model, fname):
    with open(fname, 'wb') as f:
        torch.save(model.state_dict(), fname)


def model_load(model, fname):
    model.load_state_dict(torch.load(fname))
    return model


def get_args():
    parser = argparse.ArgumentParser(description='NiP, lstm + aspect based sentiment analysis')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--ntokens', type=int, default=195158, help='number of tokens')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--nclasses', type=int, default=5, help='number of classes')
    parser.add_argument('--sen_len', type=int, default=600, help='maximum seq length')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--device', default='cuda', help='device to use(cpu, cuda). Default is cuda')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--resume', type=str, default='', help='path of model to resume')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument(
        '--wdecay', type=float, default=0, help='weight decay applied to all weights')
    parser.add_argument(
        '--when',
        nargs="+",
        type=int,
        default=[-1],
        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print('WARNING: cuda is not available in your device. Running on CPU')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
