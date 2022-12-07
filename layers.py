import torch.nn.functional as F
from torch import nn, optim
import torch
import math


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = feature_dim
        self.Wk = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.Wq = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.Wp = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.Wk.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.Wk.bias)
        nn.init.xavier_uniform_(self.Wq.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.Wq.bias)
        nn.init.xavier_uniform_(self.Wp.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # query   : [batch_size, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query=None, mask=None):
        if query is None:
            attention = self.Wp(torch.tanh(self.Wk(feature)))
        else:
            attention = self.Wp(torch.tanh(self.Wk(feature) + self.Wq(query.unsqueeze(1))))
            # attention = torch.matmul(self.Wk(feature), self.Wq(query).unsqueeze(-1)) / math.sqrt(self.hidden_size)
            # attention = torch.matmul(torch.tanh(self.Wk(feature)), (self.Wp.weight + self.Wq(query)).unsqueeze(-1))
            # a = attention.squeeze(dim=2)

        a = attention.squeeze(dim=2)

        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)  # [batch_size, feature_dim]
        return out


class SelfDotAttention(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfDotAttention, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout

        self.pos_embedding = nn.Embedding(200, self.dim)

        self.affine1 = nn.Linear(self.dim, self.dim)
        self.affine2 = nn.Linear(self.dim, 1)

    def initialize(self):
        nn.init.uniform_(self.pos_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, h, mask=None, return_logits=False, position=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        # a = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        batch_size = mask.shape[0]
        max_len = mask.shape[1]

        if position:
            pos_emb = self.pos_embedding.weight[:max_len]  # [L, d]
            pos_emb = torch.flip(pos_emb, dims=[0])  # [L, d]
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, L, d]
            h = h + pos_emb

        a = self.affine2(torch.tanh(self.affine1(h))).squeeze(2)
        if mask is not None:
            attention = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        else:
            attention = F.softmax(a, dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        # (batch, dim)
        out = torch.matmul(attention, h).squeeze(dim=1)  # [batch_size, 1, d] -> [batch_size, feature_dim]

        if return_logits:
            return out, attention.squeeze(1)
        else:
            return out


class LastQueryAttention(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(LastQueryAttention, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout

        self.pos_embedding = nn.Embedding(200, self.dim)

        self.Wq = nn.Linear(self.dim, self.dim)
        self.Wk = nn.Linear(self.dim, self.dim)
        self.Wp = nn.Linear(self.dim, 1)

    def initialize(self):
        nn.init.uniform_(self.pos_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.Wq.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.Wq.bias)
        nn.init.xavier_uniform_(self.Wk.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.Wk.bias)
        nn.init.xavier_uniform_(self.Wp.weight)
        nn.init.zeros_(self.Wp.bias)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        # a = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        batch_size = mask.shape[0]
        max_len = mask.shape[1]

        # pos_emb = self.pos_embedding.weight[:max_len]  # [L, d]
        # pos_emb = torch.flip(pos_emb, dims=[0])  # [L, d]
        # pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, L, d]
        # h = h + pos_emb
        q = h[:, -1].unsqueeze(1)
        a = self.Wp(torch.tanh(self.Wq(q) + self.Wk(h))).squeeze(2)
        # a = self.affine2(torch.tanh(self.affine1(h))).squeeze(2)
        if mask is not None:
            attention = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        else:
            attention = F.softmax(a, dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        # (batch, dim)
        out = torch.matmul(attention, h).squeeze(dim=1)  # [batch_size, 1, d] -> [batch_size, feature_dim]

        if return_logits:
            return out, attention.squeeze(1)
        else:
            return out


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int):
        super(MultiHeadAttention, self).__init__()
        self.head_num = h
        self.head_dim = d_model // self.head_num
        self.d_model = d_model
        self.attention_scalar = math.sqrt(float(self.head_dim))
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, h * d_v]
    def forward(self, feature, mask=None):
        Q, K, V = feature, feature, feature
        batch_size = Q.size(0)
        max_len = mask.shape[1]

        Q = self.W_Q(Q).view([batch_size, max_len, self.head_num, self.head_dim])  # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, max_len, self.head_num, self.head_dim])  # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, max_len, self.head_num, self.head_dim])  # [batch_size, len_k, h, d_v]

        Q = Q.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.head_num, max_len, self.head_dim])  # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.head_num, max_len, self.head_dim])  # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.head_num, max_len, self.head_dim])  # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.head_num]).view([batch_size * self.head_num, 1, max_len]).repeat(
                [1, max_len, 1])  # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(_mask == 0, -1e9), dim=2)  # [batch_size * h, len_q, len_k]
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view(
            [batch_size, self.head_num, max_len, self.head_dim])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view([batch_size, max_len, -1])  # [batch_size, len_q, h * d_v]
        return out