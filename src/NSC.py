import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

def masked_softmax(logits, mask, dim=1, epsilon=1e-9):
    """ logits, mask has same size """
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    max_logits = torch.max(masked_logits, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_logits - max_logits)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


def masked_average(value, mask, eps=1e-9):
    """
    value, mask has same size except final dimension
    and mask has to exactly one less rank than value.
    Dimension for aggragation is mask.dim()-1
    ex)
    - value : N, T, D
    - mask  : N, T
    - return: N, D
    """
    mask = mask.float()
    dim = mask.dim() - 1  # ex) 1
    S = (value * mask.unsqueeze(dim=dim + 1)).sum(dim=dim)  # ex) N, D
    N = mask.sum(dim=dim)  # ex) N
    return S / (N.unsqueeze(dim=dim) + eps)  # ex) N, D


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

    def forward(self, v, mask):
        """
                :param: v: (N1, N2, ..., NN), L, Dv
                :param: mask: (N1, N2, ..., NN), L
                We perform attention over "L"
                :return: v: (N1, N2, ..., NN), Dv ; weighted averaged vector
                :return: a: (N1, N2, ..., NN), L  ; weights for vectors (0<=ai<=1, sum to 1)
                """
        z = self.W(v).squeeze(dim=-1)  # (N1, N2, ..., NN), L
        a = masked_softmax(logits=z, mask=mask, dim=-1)
        # a: (N1, N2, ..., NN), L
        NNs, L, D = v.shape[:-2], v.shape[-2], v.shape[-1]
        return torch.bmm(
            a.view(np.prod(NNs), L).unsqueeze(dim=-2),  # (N1*N2*...*NN), 1, L
            v.view(np.prod(NNs), L, D)  # (N1*N2*...*NN), L, D
        ).squeeze(  # (N1*N2*...*NN), 1, D
            dim=-2  # (N1*N2*...*NN), D
        ).view(*NNs, -1)  # (N1, N2, ..., NN), D


class MLPAttentionWithQuery(nn.Module):
    def __init__(self, Dv, Dq):
        """ ev_t: encoded_vecs, q_t: query
        u_t = tanh(W*(ev_t, q_t)+b)
        a_t = softmax(v^T u_t)
        """
        super().__init__()
        self.W = nn.Sequential(
            nn.Linear(
                Dv + Dq,
                Dv
            ),
            nn.Tanh(),
            nn.Linear(Dv, 1, bias=False)
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, v, q, mask=None):
        """
        :param: v: (N1, N2, ..., NN), L, Dv
        :param: q: (N1, N2, ..., NN), L, Dq
        :param: mask: (N1, N2, ..., NN), L
        We perform attention over "L"
        :return: v: (N1, N2, ..., NN), Dv ; weighted averaged vector
        :return: a: (N1, N2, ..., NN), L  ; weights for vectors (0<=ai<=1, sum to 1)
        """
        z = self.W(torch.cat([v, q], dim=-1)).squeeze(dim=-1)  # (N1, N2, ..., NN), L
        if mask is None:
            a = F.softmax(z, dim=-1)
        else:
            a = masked_softmax(logits=x, mask=mask, dim=-1)
        # a: (N1, N2, ..., NN), L
        if self.attention_only:
            return attention
        else:
            NNs, L, D = v.shape[:-2], v.shape[-2], v.shape[-1]
            return (
                torch.bmm(
                    a.view(np.prod(NNs), L).unsqueeze(dim=-2),  # (N1*N2*...*NN), 1, L
                    v.view(np.prod(NNs), L, D)  # (N1*N2*...*NN), L, D
                ).squeeze(  # (N1*N2*...*NN), 1, D
                    dim=-2  # (N1*N2*...*NN), D
                ).view(*NNs, -1)  # (N1, N2, ..., NN), D
                , a  # (N1, N2, ..., NN), L
            )


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
        return output, (hn, cn)


class AttributeEmb(nn.Module):
    def __init__(self, args, attribute_dim):
        super().__init__()
        self.args = args
        self.user_emb = nn.Sequential(
            nn.Embedding(self.args.num_user, attribute_dim),
            nn.Dropout(self.args.attribute_dropout)
        )
        self.product_emb = nn.Sequential(
            nn.Embedding(self.args.num_product, attribute_dim),
            nn.Dropout(self.args.attribute_dropout)
        )

    def forward(self, user, product):
        return torch.cat(
            self.user_emb(user),
            self.product_emb(product),
            -1)  # (N, N1, ..., NN), 2*Da


class NSC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.WordEmb = nn.Embedding(self.args.ntokens, self.args.emsize)
        self.BiLSTM = TextLSTM(
            input_size=self.args.emsize, hidden_size=self.args.nhid // 2,
            bidirectional=True, bias=True, num_layers=self.args.nlayers,
            device=self.args.device)
        # self.AttributeEmb = AttributeEmb(args, attribute_dim=self.args.attribute_dim, hid_dim=0, Ha=False)
        # self.AttentionLayer = MLPAttentionWithQuery(
        # 	encoder_dim=self.args.hidden_dim,
        # 	query_dim=3*self.args.attribute_dim,
        # 	device=self.args.device,
        # 	)
        self.AttentionLayer = MLPAttentionWithoutQuery(
            self.args.nhid,
            # device=self.args.device,
        )

        self.Classifier = nn.Linear(self.args.nhid, 5)

    def forward(self,
                text, length, mask,
                # user, product,
                ):
        """
        :param: user, product: N
        :param: text, mask: N, L
        :param: length: N,
        """
        N, L = text.shape
        x = self.WordEmb(text)  # N, L, Dw
        x = self.BiLSTM(inputs=x, length=length)[0]  # N, L, Dh

        # Expand Query Attributes
        # user = user.unsqueeze(-1).repeat(1, L) # N, L
        # product = product.unsqueeze(-1).repeat(1, L) # N, L
        # query = self.AttributeEmb(user, product) # N, L, 2*Da
        # x = self.AttentionLayer(v=x, q=query, mask=mask)[0] # N, Dh
        x = self.AttentionLayer(v=x, mask=mask)  # N, Dh
        return self.Classifier(x)  # N, 5
