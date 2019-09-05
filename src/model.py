import torch.nn as nn

class Model(nn.Module):
    def __init__(self, inp_dim, emb_dim, hidden_dim, nlayers, output_dim, sen_len):
        super(Model, self).__init__()
        self.emb = nn.Embedding(inp_dim, emb_dim)
        self.lstm1 = nn.LSTM(emb_dim, hidden_dim)
        self.avg1 = nn.AvgPool1d(3, padding=3)
        # Sentence sort layer
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.avg1 = nn.AvgPool1d(3, padding=3)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.ninp = inp_dim
        self.nhid = hidden_dim
        self.nlayers = nlayers
        self.tie_weights = True
        self.max_sen_len = sen_len

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
