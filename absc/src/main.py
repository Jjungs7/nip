import argparse
from train import Instructor


def main(args):
    instr = Instructor(args)
    if not args.test:
        if args.category == 'abae':
            instr.abae_train()
        elif args.category == 'absc':
            instr.train()
    else:
        if args.category == 'abae':
            instr.abae_test()
        elif args.category == 'absc':
            instr.test()


def get_args():
    parser = argparse.ArgumentParser(description='NiP, lstm + aspect based sentiment analysis')
    parser.add_argument('--naspects', type=int, default=5, help='the number of aspects')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--nhid', type=int, default=300, help='number of hidden dimension')
    parser.add_argument('--nclasses', type=int, default=5, help='number of classes')
    parser.add_argument('--sen_len', type=int, default=600, help='maximum seq length')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--reg', type=float, default=0.1, help='regularization term')
    parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--abae_path', type=str, default='../abae', help='abae model path')
    parser.add_argument('--absc_path', type=str, default='../absc', help='absc model path')
    parser.add_argument('--data_path_prefix', type=str, default='../../data/absc_encoded', help='encoded data')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
    parser.add_argument('--device', default='cuda', help='device to use(cpu, cuda). Default is cuda')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--category', type=str, default='', help='which model? default(nothing) or abae or absc')
    parser.add_argument('--model_name', type=str, default='', help='model name to invoke')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--tokenized_path', type=str, default='../../data/tokenized.txt', help='data to word2vec')
    parser.add_argument('--w2v_path', type=str, default='../word_vector/yelp.w2v', help='data to word2vec')
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
