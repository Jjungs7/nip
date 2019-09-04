from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

ENCO_PATH = '../data/encoded'
DATA_PATH = '../data'

train_name = 'train.txt'
dev_name = 'dev.txt'
test_name = 'test.txt'

seq_len = 600
batch_size = 32


def split_review_score(name):
    with open(os.path.join(ENCO_PATH, name), 'r') as f:
        lines = f.readlines()

    review_list = []
    score_list = []

    print("Processing " + name + "...")
    for i, line in enumerate(tqdm(lines)):
        review = line.split('\t')[0]
        score = int(line.split('\t')[1])
        review_list.append(review.split(' '))
        score_list.append(score)

    return review_list, score_list


def check_len(name, review_list):
    reviews_len = [len(x) for x in review_list]
    plt.hist(reviews_len, bins=50)
    plt.show()

    num = 0
    for l in reviews_len:
        if l <= seq_len:
            num += 1

    print(num)


def remove_outlier(review_list, score_list):
    new_review_list = []
    new_score_list = []
    for i, review in enumerate(review_list):
        if len(review) <= 600:
            new_review_list.append(review)
            new_score_list.append(score_list[i])

    return new_review_list, new_score_list


def pad_and_trun(review_list):
    features = np.zeros((len(review_list), seq_len), dtype=int)

    print("Processing padding / truncating...")
    for i, review in enumerate(tqdm(review_list)):
        review_len = len(review)

        if review_len <= seq_len:
            zeroes = list(np.zeros(seq_len-review_len, dtype=int))
            new = zeroes + review
            features[i, :] = np.array(new)
        elif review_len >= seq_len:
            new = review[:seq_len]
            features[i, :] = np.array(new)

    return features


def get_loader(name):
    review_list, score_list = split_review_score(name)
    review_list, score_list = remove_outlier(review_list, score_list)

    review_trimmed_list = pad_and_trun(review_list)

    score_list = np.array(score_list)
    data = TensorDataset(torch.from_numpy(review_trimmed_list), torch.from_numpy(score_list))
    dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)

    return dataloader
