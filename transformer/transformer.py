import torch
from .attention import MultiHeadAttention

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, external_dim, internal_dim):
        super(FeedForwardNetwork, self).__init__()
        self.external_dim = external_dim
        self.internal_dim = internal_dim

        self.fc1 = torch.nn.Linear(self.external_dim, self.internal_dim)
        self.fc2 = torch.nn.Linear(self.internal_dim, self.external_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class TransformerEncoderCell(torch.nn.Module):
    def __init__(self, dim, n_heads, ffn_dim=256):
        super(TransformerEncoderCell, self).__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.n_heads = n_heads

        self.mha = MultiHeadAttention(self.dim, self.n_heads)
        self.ln_mha = torch.nn.LayerNorm(self.dim)
        self.ffn = FeedForwardNetwork(self.dim, self.ffn_dim)
        self.ln_ffn = torch.nn.LayerNorm(self.dim)
    
    def forward(self, x):
        _x, att = self.mha(q=x, k=x, v=x)
        x = x + _x
        x = self.ln_mha(x)
        _x = self.ffn(x)
        x = x + _x
        x = self.ln_ffn(x)
        return x, att

class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim, n_heads, n_layers=1, ffn_dim=256):
        super(TransformerEncoder, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim

        self.te_cells = []
        for _ in range(0, self.n_layers):
            self.te_cells.append(TransformerEncoderCell(self.dim, self.n_heads, ffn_dim=self.ffn_dim))
        self.te_cells = torch.nn.ModuleList(self.te_cells)
    
    def forward(self, x):
        atts = []
        for idx in range(0, self.n_layers):
            x, _att = self.te_cells[idx](x)
            atts.append(_att)
        return x, atts