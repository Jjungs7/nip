import torch.nn as nn

class Model(nn.Module):
    def __init__(self, inp_dim, emb_dim, output_dim, max_sen_len):
        super(Model, self).__init__()
        self.emb = nn.Embedding(inp_dim, emb_dim)
        self.lstm1 = nn.LSTM(emb_dim, emb_dim)
        self.avg1 = nn.AvgPool1d(emb_dim)
        # Sentence sort layer
        self.lstm2 = nn.LSTM(emb_dim, emb_dim)
        self.avg2 = nn.AvgPool1d(emb_dim)
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, output_dim)

        self.max_sen_len = max_sen_len

    def forward(self, inp, hidden):
        inp = self.emb(inp)
        inp, hidden = self.lstm1(inp, hidden)
        inp = self.avg1(inp)

        # Sentence sort
        [sen_len, emb_len] = inp.shape
        inp = inp.reshape((sen_len / self.max_sen_len, self.max_sen_len, emb_len))
        inp = inp.permute((1, 0, 2))

        inp, hidden = self.lstm2(inp, hidden)
        inp = self.avg2(inp)
        inp = self.fc1(inp)
        inp = self.fc2(inp)
        return inp, hidden

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return [(weight.new_zeros(1, batch_size, self.nhid if l != self.nlayers - 1 else
        (self.ninp if self.tie_weights else self.nhid)),
                 weight.new_zeros(1, batch_size, self.nhid if l != self.nlayers - 1 else
                 (self.ninp if self.tie_weights else self.nhid)))
                for l in range(self.nlayers)]
