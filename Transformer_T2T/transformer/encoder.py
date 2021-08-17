import torch
from .attention import MultiHeadAttention, FeedForwardNetwork

class EncoderLayer(torch.nn.Module):
  def __init__(
    self,
    hidden_dim,
    n_heads,
    filter_size,
    dropout_att_rate=0.1,
    dropout_relu_rate=0.1,
    dropout_postprocess_rate=0.2
  ):
    super(EncoderLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_heads = n_heads
    self.filter_size = filter_size
    self.dropout_att_rate = dropout_att_rate
    self.dropout_relu_rate = dropout_relu_rate
    self.dropout_postprocess_rate = dropout_postprocess_rate

    self.preprocess_att = torch.nn.LayerNorm(hidden_dim)
    self.attention = MultiHeadAttention(
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout_rate=dropout_att_rate
                      )
    self.dropout_postprocess = torch.nn.Dropout(1 - dropout_postprocess_rate)
    self.preprocess_ffn = torch.nn.LayerNorm(hidden_dim)
    self.ffn = FeedForwardNetwork(hidden_dim, filter_size, dropout_relu_rate)

  def forward(self, x, bias=None):
    y = self.preprocess_att(x)
    y, weights = self.attention(y, y, y, bias)
    x = self.dropout_postprocess(y) + x

    y = self.preprocess_ffn(x)
    y = self.ffn(y)
    x = self.dropout_postprocess(y) + x

    return x, weights

class Encoder(torch.nn.Module):
  def __init__(
    self,
    n_layers,
    hidden_dim,
    n_heads,
    filter_size,
    dropout_att_rate=0.1,
    dropout_relu_rate=0.1,
    dropout_postprocess_rate=0.2
  ):
    super(Encoder, self).__init__()
    self.n_layers = n_layers
    encoders = [
      EncoderLayer(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        filter_size=filter_size,
        dropout_att_rate=0.1,
        dropout_relu_rate=0.1,
        dropout_postprocess_rate=0.2
      )
    for _ in range(0, n_layers)]
    self.encoders = torch.nn.ModuleList(encoders)
    self.preprocess_out = torch.nn.LayerNorm(hidden_dim)

  def forward(self, x, bias=None):
    attention_list = []
    for encoder in self.encoders:
      x, weights = encoder(x, bias)
      attention_list.append(weights)
    x = self.preprocess_out(x)
    return x, attention_list