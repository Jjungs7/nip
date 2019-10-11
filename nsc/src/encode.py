import os
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPParser
import numpy as np

tokenizer = CoreNLPParser()

SAVE_PATH = '../data'
CONV_PATH = '../data/converted'
ENCO_PATH = '../data/encoded'

train_name = 'train.txt'
dev_name = 'dev.txt'
test_name = 'test.txt'
vocab_name = 'vocab.txt'


def split_review_score(name):
    with open(os.path.join(CONV_PATH, name), 'r') as f:
        lines = f.readlines()

    review_list = []
    score_list = []

    print("Processing " + name + "...")
    for i, line in enumerate(tqdm(lines)):
        review = line.split('\t')[0]
        score = int(line.split('\t')[1])
        review_list.append(review)
        score_list.append(score)

    return review_list, score_list


def make_dict():
    with open(os.path.join(SAVE_PATH, vocab_name), 'r') as f:
        vocab_list = f.read().splitlines()

    vocab_to_int = {w: i+1 for i, w in enumerate(vocab_list)}
    int_to_vocab = {i+1: w for i, w in enumerate(vocab_list)}

    return vocab_to_int, int_to_vocab


def encode_review_list(name, review_list, score_list, vocab_to_int):
    print("Encoding review list...")
    with open(os.path.join(ENCO_PATH, name), 'w') as f:
        for i, review in enumerate(tqdm(review_list)):
            tokens = tokenizer.tokenize(review)
            token_list = [str(vocab_to_int[token]) for token in tokens]
            encoded_text = ' '.join(token_list)
            f.write(encoded_text + '\t' + str(score_list[i]) + '\n')


def encode_score_list(score_list):
    return np.array(score_list)


if __name__=="__main__":
    try:
        os.mkdir(ENCO_PATH)
    except FileExistsError:
        print("data/encoded folder already exists. Continue?(y/n)")
        cont = input()
        if cont == "n" or cont == "N":
            exit(0)

    train_review_list, train_score_list = split_review_score(train_name)
    dev_review_list, dev_score_list = split_review_score(dev_name)
    test_review_list, test_score_list = split_review_score(test_name)

    vocab_to_int, int_to_vocab = make_dict()

    encode_review_list(train_name, train_review_list, train_score_list, vocab_to_int)
    encode_review_list(dev_name, dev_review_list, dev_score_list, vocab_to_int)
    encode_review_list(test_name, test_review_list, test_score_list, vocab_to_int)


