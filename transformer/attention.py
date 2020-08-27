import torch
import math

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.scale = math.sqrt(self.dim) * 10
    
    def forward(self, q, k, v, additive_mask=None):
        x = torch.bmm(q, torch.transpose(k, 1, 2)) / self.scale
        if additive_mask is not None:
            x = x + additive_mask
        x = torch.softmax(x, dim=-1) + 1e-5
        # x = torch.sigmoid(x) + 1e-5
        att = x
        x = torch.bmm(x, v)
        return x, att

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.fc_q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.fc_k = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.fc_v = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.sdpas = []
        for _ in range(0, self.n_heads):
            self.sdpas.append(ScaledDotProductAttention(self.dim))
        self.sdpas = torch.nn.ModuleList(self.sdpas)
        self.fc_output = torch.nn.Linear(self.dim*self.n_heads, self.dim, bias=True)
    
    def forward(self, q, k, v):
        x_head = []
        att_head = []
        for idx in range(0, self.n_heads):
            _q, _k, _v = self.fc_q(q), self.fc_k(q), self.fc_v(q)
            _x, _att = self.sdpas[idx](_q, _k, _v)
            x_head.append(_x)
            att_head.append(_att)
        x_head = torch.cat(x_head, dim=-1)
        x = self.fc_output(x_head)
        att_head = torch.stack(att_head, dim=1)
        return x, att_head