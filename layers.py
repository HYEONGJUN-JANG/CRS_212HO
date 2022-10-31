import torch.nn.functional as F
from torch import nn, optim
import torch
import math


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = feature_dim
        self.linear_key = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.linear_query = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.linear_proj = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.linear_key.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.linear_key.bias)
        nn.init.xavier_uniform_(self.linear_query.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.linear_query.bias)
        nn.init.xavier_uniform_(self.linear_proj.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # query   : [batch_size, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query=None, mask=None):
        if query is None:
            attention = torch.tanh(self.linear_key(feature))  # [batch_size, length, attention_dim]
            a = self.linear_proj(attention).squeeze(dim=2)  # [batch_size, length]

        else:
            # attention = torch.tanh(
            #     self.linear_key(feature) + self.linear_query(query).unsqueeze(1))  # [batch_size, length, attention_dim]
            # a = self.linear_proj(attention).squeeze(dim=2)  # [batch_size, length]
            # query = query
            a = torch.bmm(self.linear_key(feature), self.linear_query(query).unsqueeze(-1)).squeeze(dim=2)
            a = a / math.sqrt(self.hidden_size)

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
