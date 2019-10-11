# -*- coding: utf-8 -*-

import codecs
import sys

import gensim
from tqdm import tqdm

w2v_name = 'yelp'

class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in tqdm(codecs.open(self.filename, "r", "utf-8")):
            yield line.strip().split()


def main(path):
    sentences = Sentences(path)
    model = gensim.models.Word2Vec(sentences, size=400, window=8, min_count=10, workers=4, sg=1, hs=0,
                                   negative=10, iter=3, max_vocab_size=200000)
    model.save("../word_vectors/" + w2v_name + ".w2v")
    # absc.wv.save_word2vec_format("word_vectors/" + domain + ".txt", binary=False)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "../data/tokenized.txt"

    try:
        import os
        os.mkdir("../word_vectors/")
    except:
        pass

    print("Training w2v on dataset", path)

    main(path)

    print("Training done.")

    model = gensim.models.Word2Vec.load("../word_vectors/" + w2v_name + ".w2v")

    for word in ["I", "love", "taste", "service", "price"]:
        if word in model.wv.vocab:
            print(word, [w for w, c in model.wv.similar_by_word(word=word)])
        else:
            print(word, "not in vocab")
