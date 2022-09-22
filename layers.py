import torch.nn.functional as F
from torch import nn, optim
import torch


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(AdditiveAttention, self).__init__()
        self.affine1 = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))  # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)  # [batch_size, length]
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

        self.affine1 = nn.Linear(self.dim, self.dim)
        self.affine2 = nn.Linear(self.dim, 1)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        # a = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
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
