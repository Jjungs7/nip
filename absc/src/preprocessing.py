import os
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPParser

tokenizer = CoreNLPParser()

DATA_PATH = '../data'

train_name = 'train.txt'
dev_name = 'dev.txt'
test_name = 'test.txt'
tokenized_name = 'tokenized.txt'

sentence_list = []


def get_review(line):
    review = line.split('\t')
    return review[-1]


def get_sentences(name):
    with open(os.path.join(DATA_PATH, name), 'r') as f:
        lines = f.readlines()
        print(f"Spliting sentences in {name}...")
        for line in tqdm(lines):
            review = get_review(line)
            review = review.split('<sssss>')
            for sentence in review:
                sentence_list.append(sentence.strip())


def tokenize_sentences():
    print("Tokenizing all sentences...")
    with open(os.path.join(DATA_PATH, tokenized_name), 'w') as f:
        for sent in tqdm(sentence_list):
            tokens = tokenizer.tokenize(sent)
            line = ' '.join(tokens)
            line.replace('  ', ' ')
            f.write(line.strip() + '\n')


if __name__=="__main__":
    get_sentences(train_name)
    get_sentences(dev_name)
    get_sentences(test_name)

    tokenize_sentences()


