import argparse
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from model import Model
import numpy as np
import time


def main(args):
    model = Model(args.ntokens, args.emsize, args.nclasses, args.sen_len)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    if args.resume:
        model = model_load(model, args.resume)

    # Dataset
    # Train
    # Eval


def train(model, data, optimizer, loss_fn, args):
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)

    #Train


def evaluate(model, data, optimizer, loss_fn, args):
    total_loss = 0
    model.eval()
    hidden = model.init_hidden(args.batch_size)

    # Evaluate


def model_save(model, fname):
    with open(fname, 'wb') as f:
        torch.save(model.state_dict(), fname)


def model_load(model, fname):
    model.load_state_dict(torch.load(fname))
    model.eval()
    return model


def get_args():
    parser = argparse.ArgumentParser(description='NiP, lstm + aspect based sentiment analysis')
    parser.add_argument(
        '--data', type=str, default='penn_treebank_dataset', help='the name of the dataset to load')
    parser.add_argument(
        '--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--ntokens', type=int, default=400, help='number of tokens')
    parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--nclasses', type=int, default=10, help='number of classes')
    parser.add_argument('--sen_len', type=int, default=10, help='maximum sentence length')
    parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--nonmono', type=int, default=5, help='random seed')
    parser.add_argument('--device', default='cuda', help='device to use(cpu, cuda). Default is cuda')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--resume', type=str, default='', help='path of model to resume')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use (sgd, adam)')
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
