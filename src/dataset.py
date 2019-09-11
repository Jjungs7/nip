import os, numpy as np, pickle, torch, random, itertools
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm


def text_padding(text, length, return_mask=False):
    """
    text: list of token indices
    length: list of length of tokens (same size with text)
    return: padded text tokens, ndarray, np.int64
    """
    maxlen = max(length)
    num_data = len(text)
    if return_mask:
        mask = np.zeros((num_data, maxlen), dtype=np.uint8)
    padded_sentences = np.zeros((num_data, maxlen), dtype=np.int64)
    if return_mask:
        for i, (l, x) in enumerate(zip(length, text)):
            padded_sentences[i][:l] = x
            mask[i][:l] = 1
        return padded_sentences, mask
    else:
        for i, (l, x) in enumerate(zip(length, text)):
            padded_sentences[i][:l] = x
        return padded_sentences


class NIPDataset(Dataset):
    def __init__(self, args, data_path):
        super().__init__(args, data_path)
        with open(self.data_path,'r') as f:
            lines = f.readlines()

        self.data = []

        for i, line in enumerate(tqdm(lines)):
            review = line.split('\t')[0]
            score = int(line.split('\t')[1])
            review = review.split(' ')
            self.data.append((score, review, len(review)))

    def custom_collate_fn(self, sample_batch):
        rating = []
        text = []
        length = []
        for i, element in enumerate(self.data):
            rating.append(element[0])
            text.append(element[1])
            length.append(element[2])

        # to Tensor
        rating = torch.LongTensor(rating).to(self.args.device)  # N
        text, mask = text_padding(text, length, return_mask=True)
        text = torch.from_numpy(text).to(self.args.device)  # N, L
        length = torch.LongTensor(length).to(self.args.device)  # N
        mask = torch.from_numpy(mask).to(self.args.device)  # N, L

        return (text, length, mask, rating)

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]