import numpy as np, torch, torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import normalize, softmax
from word2vec import word2vec

class MLPAttentionWithoutQuery(nn.Module):
    def __init__(self, Dv):
        super().__init__()
        self.W = nn.Sequential(
            nn.Linear(Dv, Dv, bias=False),
            nn.Tanh(),
            nn.Linear(Dv, 1, bias=False)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.constant_(p)

    def masked_softmax(self, logits, mask, dim=1, epsilon=1e-9):
        """ logits, mask has same size """
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        max_logits = torch.max(masked_logits, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_logits - max_logits)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return masked_exps / masked_sums

    def forward(self, v, mask):
        """
        :param: v: (N1, N2, ..., NN), L, Dv
        :param: mask: (N1, N2, ..., NN), L
        We perform attention over "L"
        :return: v: (N1, N2, ..., NN), Dv ; weighted averaged vector
        :return: a: (N1, N2, ..., NN), L  ; weights for vectors (0<=ai<=1, sum to 1)
        """
        z = self.W(v).squeeze(dim=-1)  # (N1, N2, ..., NN), L
        a = self.masked_softmax(logits=z, mask=mask, dim=-1)
        # a: (N1, N2, ..., NN), L
        NNs, L, D = v.shape[:-2], v.shape[-2], v.shape[-1]
        return torch.bmm(
            a.view(np.prod(NNs), L).unsqueeze(dim=-2),  # (N1*N2*...*NN), 1, L
            v.view(np.prod(NNs), L, D)  # (N1*N2*...*NN), L, D
        ).squeeze(  # (N1*N2*...*NN), 1, D
            dim=-2  # (N1*N2*...*NN), D
        ).view(*NNs, -1)  # (N1, N2, ..., NN), D


class TextLSTM(nn.Module):
    def __init__(self,
                 input_size, hidden_size,
                 num_layers=1, bidirectional=False, dropout=0, bias=True,
                 batch_first=True,
                 device='cpu'):
        super(TextLSTM, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        if self.num_layers == 1: dropout = 0
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout
        )

    def zero_init(self, batch_size):
        nd = 1 if not self.bidirectional else 2
        h0 = Variable(torch.zeros((self.num_layers * nd, batch_size, self.hidden_size))).to(self.device)
        c0 = Variable(torch.zeros((self.num_layers * nd, batch_size, self.hidden_size))).to(self.device)
        return (h0, c0)

    def forward(self, inputs, length, rnn_init=None, is_sorted=False):
        if rnn_init is None:
            rnn_init = self.zero_init(inputs.size(0))
        if not is_sorted:
            sort_idx = torch.sort(length, descending=True)[1]
            inputs = inputs[sort_idx]
            length = length[sort_idx]
            h0, c0 = rnn_init
            rnn_init = (h0[:, sort_idx, :], c0[:, sort_idx, :])
            unsort_idx = torch.sort(sort_idx)[1]
        x_pack = nn.utils.rnn.pack_padded_sequence(inputs, length, batch_first=self.batch_first)
        output, (hn, cn) = self.rnn(x_pack, rnn_init)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        if not is_sorted:
            output = output[unsort_idx]
            hn = hn[:, unsort_idx, :]
            cn = cn[:, unsort_idx, :]
        return output, (hn, cn)


class abae_attention(nn.Module):
    def __init__(self, d_embed):
        super(abae_attention, self).__init__()
        self.M = nn.Linear(d_embed, d_embed)
        self.M.weight.data.uniform_(-0.1, 0.1) # -0.1과 0.1 사이의 값으로 초기화

    def forward(self, e_i):
        y_s = torch.mean(e_i, dim=-1)
        d_i = torch.bmm(e_i.transpose(1, 2), self.M(y_s).unsqueeze(2)).tanh()
        a_i = d_i / sum(torch.exp(d_i))
        return a_i.squeeze(1)


class ABAE(nn.Module):
    def __init__(self, E, T):
        super(ABAE, self).__init__()
        n_vocabs, d_embed = E.shape
        n_aspects, d_embed = T.shape
        self.E = nn.Embedding(n_vocabs, d_embed)
        self.T = nn.Embedding(n_aspects, d_embed)
        self.attention = abae_attention(d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)
        self.T.weight = nn.Parameter(torch.from_numpy(T), requires_grad=True)

    def forward(self, pos, negs):
        p_t, z_s = self.predict(pos)
        r_s = normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1)
        e_n = self.E(negs).transpose(-2, -1)
        z_n = normalize(torch.mean(e_n, dim=-1), dim=-1)
        return r_s, z_s, z_n

    def predict(self, x):
        e_i = self.E(x).transpose(1, 2)
        a_i = self.attention(e_i)
        z_s = normalize(torch.bmm(e_i, a_i).squeeze(2), dim=-1)
        p_t = softmax(self.linear(z_s), dim=1)
        return p_t, z_s

    def aspects(self):
        E_n = normalize(self.E.weight, dim=1)
        T_n = normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection


class ABSC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.w2v = word2vec(self.args.tokenized_path)
        self.w2v.embed(self.args.w2v_path, self.args.emsize)
        self.w2v.aspect(self.args.n_aspects)

        self.ABAE = ABAE(self.w2v.E, self.w2v.T)
        if self.args.device is 'cuda':
            self.ABAE = self.ABAE.cuda()
        self.BiLSTM = TextLSTM(
            input_size=self.args.emsize, hidden_size=self.args.nhid // 2,
            bidirectional=True, bias=True, num_layers=self.args.nlayers,
            device=self.args.device)
        self.AttentionLayer = MLPAttentionWithoutQuery(self.args.nhid)
        self.Classifier = nn.Linear(self.args.nhid, 5)

    def forward(self, text, length, mask):
        """
        :param: text, mask: N, L
        :param: length: N,
        """
        x = self.WordEmb(text)  # N, L, Dw



        x = self.BiLSTM(inputs=x, length=length)[0]  # N, L, Dh

        x = self.AttentionLayer(v=x, mask=mask)  # N, Dh
        return self.Classifier(x)  # N, 5
