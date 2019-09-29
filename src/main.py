import argparse
import nsc
from train import Instructor
import dataset

#vocab_to_int, int_to_vocab = encode.make_dict()
MODEL_PATH = 'model'

def main(args):
    instr = Instructor(args)
    if not args.test:
        instr.train()
    else:
        instr.test()


def get_args():
    parser = argparse.ArgumentParser(description='NiP, lstm + aspect based sentiment analysis')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--ntokens', type=int, default=195158, help='number of tokens')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--nhid', type=int, default=300, help='number of hidden dimension')
    parser.add_argument('--nclasses', type=int, default=5, help='number of classes')
    parser.add_argument('--sen_len', type=int, default=600, help='maximum seq length')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--model_save_path', type=str, default='../model', help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
    parser.add_argument('--device', default='cuda', help='device to use(cpu, cuda). Default is cuda')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--load_model', type=str, default='', help='path of model to resume')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--data_path_prefix', type=str, default='../data/encoded/', help='data to use')
    parser.add_argument('--test', type=bool, default=False, help='determine if this run is test or train')
    parser.add_argument(
        '--wdecay', type=float, default=0, help='weight decay applied to all weights')
    parser.add_argument(
        '--when',
        nargs="+",
        type=int,
        default=[-1],
        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    args = parser.parse_args()

    import torch.cuda
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print('WARNING: cuda is not available in your device. Running on CPU')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
