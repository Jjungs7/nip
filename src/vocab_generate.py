from tqdm import tqdm
from collections import Counter
import os
from nltk.parse.corenlp import CoreNLPParser

SAVE_PATH = '../data'
CONV_PATH = '../data/converted'

names = ['train.txt', 'dev.txt', 'test.txt']
review_list = []
token_list = []

tokenizer = CoreNLPParser()


def make_review_list(name):
    with open(os.path.join(CONV_PATH, name)) as f:
        lines = f.readlines()

    print("Processing " + name + "...")
    for i, line in enumerate(tqdm(lines)):
        review_list.append(line.split('\t')[0])


def make_token_list():
    print("Making token_list...")
    for i, review in enumerate(tqdm(review_list)):
        tokens = tokenizer.tokenize(review)
        for token in tokens:
            token_list.append(token)


def make_vocab():
    count_words = Counter(token_list)

    total_count = len(token_list)
    sorted = count_words.most_common(total_count)

    with open(os.path.join(SAVE_PATH, 'vocab.txt'), 'w') as f:
        for i, (w, c) in enumerate(sorted):
            f.write(w + '\n')


if __name__ == '__main__':
    for name in names:
        make_review_list(name)

    make_token_list()
    make_vocab()

