import torch
from .attention import MultiHeadAttention, FeedForwardNetwork

class DecoderLayer(torch.nn.Module):
  def __init__(
    self,
    hidden_dim,
    n_heads,
    filter_size,
    dropout_att_rate=0.1,
    dropout_relu_rate=0.1,
    dropout_postprocess_rate=0.2
  ):
    super(DecoderLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_heads = n_heads
    self.filter_size = filter_size
    self.dropout_att_rate = dropout_att_rate
    self.dropout_relu_rate = dropout_relu_rate
    self.dropout_postprocess_rate = dropout_postprocess_rate

    self.dropout_postprocess = torch.nn.Dropout(1 - dropout_postprocess_rate)
    self.preprocess_self_att = torch.nn.LayerNorm(hidden_dim)
    self.self_attention = MultiHeadAttention(
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout_rate=dropout_att_rate
                      )
    self.preprocess_cross_att = torch.nn.LayerNorm(hidden_dim)
    self.cross_attention = MultiHeadAttention(
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout_rate=dropout_att_rate
                      )
    self.preprocess_ffn = torch.nn.LayerNorm(hidden_dim)
    self.ffn = FeedForwardNetwork(hidden_dim, filter_size, dropout_relu_rate)

  # x is actually from the target embeddings.
  def forward(self, x, memory, self_bias=None, memory_bias=None):
    y = self.preprocess_self_att(x)
    y, weights_self = self.self_attention(y, y, y, self_bias)
    x = self.dropout_postprocess(y) + x

    y = self.preprocess_cross_att(x)
    y, weights_cross = self.cross_attention(y, memory, memory, memory_bias)
    x = self.dropout_postprocess(y) + x

    y = self.preprocess_ffn(x)
    y = self.ffn(y)
    x = self.dropout_postprocess(y) + x

    return x, (weights_self, weights_cross)

class Decoder(torch.nn.Module):
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
    super(Decoder, self).__init__()
    self.n_layers = n_layers
    decoders = [
      DecoderLayer(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        filter_size=filter_size,
        dropout_att_rate=0.1,
        dropout_relu_rate=0.1,
        dropout_postprocess_rate=0.2
      )
    for _ in range(0, n_layers)]
    self.decoders = torch.nn.ModuleList(decoders)
    self.preprocess_out = torch.nn.LayerNorm(hidden_dim)

  def forward(self, x, memory, self_bias=None, memory_bias=None):
    self_attention_list = []
    cross_attention_list = []
    for decoder in self.decoders:
      x, (weights_self, weights_cross) = decoder(
                                            x,
                                            memory=memory,
                                            self_bias=self_bias,
                                            memory_bias=memory_bias
                                          )
      self_attention_list.append(weights_self)
      cross_attention_list.append(weights_cross)
    x = self.preprocess_out(x)
    return x, (self_attention_list, cross_attention_list)