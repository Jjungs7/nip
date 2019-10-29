import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

import logging, colorlog
logging.disable(logging.DEBUG)
colorlog.basicConfig(
    filename=None,
    level=logging.NOTSET,
    format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
def masked_softmax(logits, mask, dim=1, epsilon=1e-9):
    """ logits, mask has same size """
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    max_logits = torch.max(masked_logits, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_logits-max_logits)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps/masked_sums
class LinearAttentionWithoutQuery(nn.Module):
    def __init__(self, encoder_dim, device=torch.device('cpu')):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.device = device
        self.z = nn.Linear(self.encoder_dim, 1, bias=False)
    def forward(self, encoded_vecs, mask=None):
        logits = self.z(encoded_vecs).squeeze(dim=2)
        if (mask is not None):
            # batch_size, max_length
            attention = masked_softmax(logits=logits, mask=mask, dim=1)
        else:
            # batch_size, max_length
            attention = F.softmax(logits, dim=1)
        return (
            torch.bmm(attention.unsqueeze(dim=1), encoded_vecs).squeeze(dim=1),
            attention
            )
class TextLSTM(nn.Module):
    def __init__(self, 
        input_size, hidden_size, 
        num_layers=1, bidirectional=False, dropout=0, bias=True, 
        batch_first=True,
        device='cpu'):
        super(TextLSTM, self).__init__()
        self.batch_first=batch_first
        self.bidirectional=bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        if self.num_layers==1: dropout=0
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
        h0 = Variable(torch.zeros((self.num_layers*nd, batch_size, self.hidden_size))).to(self.device)
        c0 = Variable(torch.zeros((self.num_layers*nd, batch_size, self.hidden_size))).to(self.device)
        return (h0, c0)
    def forward(self, inputs, length, rnn_init=None, is_sorted=False):
        if rnn_init is None:
            rnn_init = self.zero_init(inputs.size(0))
        if not is_sorted:
            sort_idx = torch.sort(length,descending=True)[1]
            inputs = inputs[sort_idx]
            length = length[sort_idx]
            # h0: size=(num_layers*bidriectional, batch_size, hidden_dim)
            # c0: size=(num_layers*bidriectional, batch_size, hidden_dim)
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
        # batch_size, length, hidden_size
        # batch_size, num_layers*bidirectional, hidden_size
        return output, (hn,cn)



class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
    def init_param(self):
        colorlog.info("[init_param] for {}".format(self.__class__.__name__))
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape)>1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.constant_(p, 0)
class CustModule(BasicModule):
    def __init__(self):
        super().__init__()
    def init_param(self):
        super().init_param()
        colorlog.info("[init_param] for {}: meta parameters: uniform_ [-0.01, 0.01]".format(self.__class__.__name__))
        for name, num_meta in self.args.meta_units:
            colorlog.info("\t {} intialized".format(name))
            nn.init.uniform_(getattr(self, name).weight, -0.01, 0.01)
class BasisCustModule(CustModule):
    def __init__(self):
        super().__init__()
        for name, num_meta in self.args.meta_units:
            setattr(self, "num_"+name, num_meta)
            # word embedding transformation parameters
            setattr(self, name, nn.Embedding(num_meta, self.args.attribute_dim))
        self.C = nn.Sequential(
            nn.Linear(self.args.attribute_dim*len(self.args.meta_units), self.args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(self.args.key_query_size, self.args.num_bases
                # , bias=False
            ), # Calculate Weights of each Basis: Key & Query Inner-product
        )
        if self.args.coef_act_fn=='none':
            self.coef_act_fn = lambda x:x
        elif self.args.coef_act_fn=='sigmoid':
            self.coef_act_fn = lambda x:torch.sigmoid(x)
        elif self.args.coef_act_fn=='tanh':
            self.coef_act_fn = lambda x:torch.tanh(x)
        elif self.args.coef_act_fn=='relu':
            self.coef_act_fn = lambda x:torch.relu(x)
        elif self.args.coef_act_fn=='softmax':
            self.coef_act_fn = lambda x:F.softmax(x, dim=-1)
        else:
            raise Exception("Unexpected activation function for mean of coefficient \"coef_act_fn:{}\"".format(self.args.coef_act_fn))
    def alpha(self, **kwargs):
        coef_logits = self.C(
            torch.cat([getattr(self, name)(idx) for name, idx in kwargs.items()], dim=1)
            ) # N, B
        return self.coef_act_fn(coef_logits) # N, B
class StochasticBasisCustModule(CustModule):
    def __init__(self):
        super().__init__()
        for name, num_meta in self.args.meta_units:
            setattr(self, "num_"+name, num_meta)
            # word embedding transformation parameters
            setattr(self, name, nn.Embedding(num_meta, self.args.attribute_dim))
        self.C = nn.Sequential(
            nn.Linear(self.args.attribute_dim*len(self.args.meta_units), self.args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(self.args.key_query_size, self.args.num_bases*2
                # , bias=False
            ), # Calculate Weights of each Basis: Key & Query Inner-product
        )
        if self.args.coef_act_fn=='none':
            self.coef_act_fn = lambda x:x
        elif self.args.coef_act_fn=='sigmoid':
            self.coef_act_fn = lambda x:torch.sigmoid(x)
        elif self.args.coef_act_fn=='tanh':
            self.coef_act_fn = lambda x:torch.tanh(x)
        elif self.args.coef_act_fn=='relu':
            self.coef_act_fn = lambda x:torch.relu(x)
        elif self.args.coef_act_fn=='softmax':
            self.coef_act_fn = lambda x:F.softmax(x, dim=-1)
        else:
            raise Exception("Unexpected activation function for mean of coefficient \"coef_act_fn:{}\"".format(self.args.coef_act_fn))
    def alpha(self, **kwargs):
        coef_logit = self.C(
            torch.cat([getattr(self, name)(idx) for name, idx in kwargs.items()], dim=1)
            ) # N, B*2 (mu and logvar)
        mu, logvar = coef_logit[:, :self.args.num_bases], coef_logit[:, self.args.num_bases:]
        std = logvar.mul(0.5).exp_() # N, B
        n_samples = self.args.n_samples if self.training else self.args.eval_n_samples
        # Sampling
        mu_repeated = mu.unsqueeze(1).repeat(1, n_samples, 1) # N, S, B
        std_repeated = std.unsqueeze(1).repeat(1, n_samples, 1) # N, S, B
        eps = torch.Tensor(std_repeated.size()).normal_().to(self.args.device) # N, S, B
        coef_sampled = eps.mul(std_repeated).add_(mu_repeated) # N, S, B
        return self.coef_act_fn(coef_sampled) # N, S, B



class BasicWordEmb(BasicModule):
    def __init__(self, args):
        super().__init__()
        self.word_em = nn.Embedding(args.vocab_size, args.word_dim, padding_idx=args._ipad)
    def forward(self, review, **kwargs):
        return self.word_em(review)


class BasicBiLSTM(BasicModule):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.LSTM = TextLSTM(
            input_size=args.word_dim,
            hidden_size=args.state_size//2, # //2 for bidirectional
            bidirectional=True,
            device=args.device
            )
    def forward(self, x, length, **kwargs):
        return self.LSTM(inputs=x, length=length)[0]


class BasicAttention(BasicModule):
    def __init__(self, args):
        super().__init__()
        self.attention = LinearAttentionWithoutQuery(
            encoder_dim=args.state_size,
            device=args.device,
            )
    def forward(self, x, mask, **kwargs):
        return self.attention(x, mask=mask)[0]


class BasicLinear(BasicModule):
    def __init__(self, args):
        super().__init__()
        self.W = nn.Linear(args.state_size, args.num_labels, bias=False)
    def forward(self, x, **kwargs):
        return self.W(x)
class BasisCustLinear(BasisCustModule):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.BasesList = nn.ModuleList([
            nn.Linear(args.state_size, args.num_labels, bias=False)
            for _ in range(args.num_bases)
        ])
    def init_param(self):
        super().init_param()
        colorlog.info("[init_param] for {}: bases for label embedding matrices: xavier_uniform_".format(self.__class__.__name__))
        for W in self.BasesList:
            nn.init.xavier_uniform_(W.weight)
    def forward(self, x, **kwargs):
        alpha = self.alpha(**kwargs)
        logits_on_bases = torch.stack([
            W(x) for W in self.BasesList
        ], dim=1) # N, B, C
        logits = torch.matmul(alpha.unsqueeze(dim=1), logits_on_bases).squeeze(1) # N, C
        return logits
class StochasticBasisCustLinear(StochasticBasisCustModule):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.BasesList = nn.ModuleList([
            nn.Linear(args.state_size, args.num_labels, bias=False)
            for _ in range(self.args.num_bases)
        ])
    def init_param(self):
        super().init_param()
        # Label Embeding Basis Initialization
        colorlog.info("[init_param] for {}: bases for label embedding matrices: xavier_uniform_".format(self.__class__.__name__))
        for W in self.BasesList:
            nn.init.xavier_uniform_(W.weight)
    def forward(self, x, **kwargs):
        coef_sampled = self.alpha(**kwargs)
        logits_on_bases = torch.stack([
            W(x) for W in self.BasesList
        ], dim=1) # N, B, C
        logits_sampled = torch.matmul(coef_sampled, logits_on_bases) # N, S, C
        return logits_sampled
    def coef_mean_and_std(self, **kwargs):
        coef_logit = self.C(
            torch.cat([getattr(self, name)(idx) for name, idx in kwargs.items()], dim=1)
            ) # N, B*2 (mu and logvar)
        mu, logvar = coef_logit[:, :self.args.num_bases], coef_logit[:, self.args.num_bases:]
        std = logvar.mul(0.5).exp_() # N, B
        return mu, std
    def mean_and_variance(self, x, **kwargs):
        coef_sampled = self.alpha(**kwargs)
        logits_on_bases = torch.stack([
            W(x) for W in self.BasesList
        ], dim=1) # N, B, C
        logits_sampled = torch.matmul(coef_sampled, logits_on_bases) # N, S, C
        return logits_sampled, coef_sampled

class BasicBias(BasicModule):
    def __init__(self, args):
        super().__init__()
        self.out_dim = args.num_labels
        self.b = nn.Parameter(torch.zeros((self.out_dim)))
    def forward(self, **kwargs):
        return self.b.unsqueeze(0)