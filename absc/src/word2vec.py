from sklearn.cluster import KMeans
import numpy as np
import gensim
import codecs
from tqdm import tqdm
import os
from nltk.parse.corenlp import CoreNLPParser

tokenizer = CoreNLPParser()


class word2vec:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.n_vocab = 0

    def __iter__(self):
        with codecs.open(self.corpus_path, 'r', 'utf-8') as f:
            for line in tqdm(f, desc='training'):
                yield line.split()

    def add_word(self, *words):
        for word in words:
            if not word in self.w2i:
                self.w2i[word] = self.n_vocab
                self.i2w[self.w2i[word]] = word
                self.n_vocab += 1

    def embed(self, model_path, d_embed):
        self.d_embed = d_embed
        if os.path.isfile(model_path):
            model = gensim.models.Word2Vec.load(model_path)
        else:
            model = gensim.models.Word2Vec(self,
                size=d_embed, max_vocab_size=200000,
                window=8, min_count=10, workers=4, sg=1, hs=0, negative=10, iter=3)
            model.save(model_path)
            model = gensim.models.Word2Vec.load(model_path)
        self.i2w, self.w2i = {}, {}
        self.E = []
        n = len(model.wv.vocab)
        for word in sorted(model.wv.vocab):
            self.add_word(word)
            self.E.append(list(model.wv[word]))
        else:
            self.add_word('<unk>')
            self.E.append([0] * d_embed)
        self.E = np.asarray(self.E).astype(np.float32)
        return self

    def aspect(self, n_aspects):
        self.n_aspects = n_aspects
        km = KMeans(n_clusters=n_aspects, random_state=0)
        km.fit(self.E)
        self.T = km.cluster_centers_.astype(np.float32)
        self.T /= np.linalg.norm(self.T, axis=-1, keepdims=True)
        return self


if __name__=='__main__':
    w2v = word2vec('../../data/tokenized.txt')
    w2v.embed('../word_vector/yelp.w2v', 400)
    print("Word2Vec training finished.")

    origin_path = '../../data/converted'
    dest_path = '../../data/absc_encoded'
    names = ['train.txt', 'dev.txt', 'test.txt']

    for name in names:
        print(f"Processing {name}...")
        with open(os.path.join(origin_path, name), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(dest_path, name), 'w') as f:
            for line in tqdm(lines):
                review = line.split('\t')[0]
                score = line.split('\t')[1]
                tokens = tokenizer.tokenize(review)
                indices = [str(w2v.w2i[token]) if token in w2v.w2i else str(w2v.w2i['<unk>']) for token in tokens]
                f.write(' '.join(indices) + '\t' + score)
