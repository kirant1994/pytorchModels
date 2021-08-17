import torch

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, hidden_dim, output_dim, n_heads, dropout_rate=0.1):
    super(MultiHeadAttention, self).__init__()
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.n_heads = n_heads
    self.dropout_rate = dropout_rate

    self.q_transform = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.k_transform = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.v_transform = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.dropout = torch.nn.Dropout(1 - dropout_rate)
    self.fc_out = torch.nn.Linear(hidden_dim, output_dim, bias=True)
  
  def forward(self, q, k, v, bias=None):
    N, T_q, D = q.shape
    _, T_k, _ = k.shape
    n_heads = self.n_heads
    assert D % n_heads == 0

    D_per_head = D // n_heads

    q = self.q_transform(q)
    k = self.k_transform(k)
    v = self.v_transform(v)

    q = q.view((N, T_q, n_heads, D_per_head))
    k = k.view((N, T_k, n_heads, D_per_head))
    v = v.view((N, T_k, n_heads, D_per_head))

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scaling_factor = (D_per_head ** (-0.5))
    q = q * scaling_factor
    logits = torch.matmul(q, k.transpose(2, 3))
    if bias is not None:
      logits += bias
    weights = torch.nn.functional.softmax(logits, dim=-1)
    weights = self.dropout(weights)

    x = torch.matmul(weights, v)
    x = x.transpose(1, 2)
    x = x.contiguous().view((N, T_q, D))
    x = self.fc_out(x)
    
    return x, weights

class FeedForwardNetwork(torch.nn.Module):
  def __init__(self, hidden_dim, filter_size, dropout_rate=0.1):
    super(FeedForwardNetwork, self).__init__()
    self.hidden_dim = hidden_dim
    self.filter_size = filter_size
    self.dropout_rate = dropout_rate

    self.fc1 = torch.nn.Linear(hidden_dim, filter_size)
    self.dropout = torch.nn.Dropout(1 - dropout_rate)
    self.fc2 = torch.nn.Linear(filter_size, hidden_dim)

  def forward(self, x):
    x = self.fc1(x)
    x = torch.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x


# Mask for the encoder. Generally non-causal.
def prepare_padding_mask(lens):
  N = len(lens)
  T = max(lens)
  mask = torch.zeros((N, T), dtype=torch.int32)
  for n, t in enumerate(lens):
    mask[n, :t] = 1
  return mask

# Mask for the causal decoder
def prepare_lower_triangular_mask(T):
  mask = torch.tril(torch.ones((T, T), dtype=torch.float64))
  return mask

def prepare_additive_lower_triangular_mask(T, N):
  mask = prepare_lower_triangular_mask(T)
  mask = (1 - mask) * -1e9
  mask = torch.stack([mask] * N, dim=0)
  return mask

def prepare_additive_padding_mask(lens):
  mask = prepare_padding_mask(lens)
  inv_mask = 1 - mask
  additive_mask = inv_mask.double() * -1e9
  return additive_mask.unsqueeze(1)

# Add positional encoding to a tensor with shape (N, T, D)
def positional_encoding(x, min_timescale=1.0, max_timescale=10000.0):
  assert len(x.shape) == 3
  N, T, D = x.shape
  assert D % 2 == 0, 'Only even dimensional signals are allowed'
  position = torch.arange(1, T+1, dtype=torch.float64)
  num_timescales = D // 2
  log_increment = torch.log(torch.tensor(max_timescale / min_timescale, dtype=torch.float64)) / (num_timescales - 1)
  inv_timescales = min_timescale * torch.exp(torch.arange(0, num_timescales, dtype=torch.float64) * -log_increment)
  position = torch.unsqueeze(position, 1)
  inv_timescales = torch.unsqueeze(inv_timescales, 0)
  scaled_time = torch.matmul(position, inv_timescales)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
  signal = signal.view((T, 2, num_timescales))
  signal = torch.transpose(signal, 1, 2)
  signal = signal.contiguous().view((T, D))

  signal = signal.unsqueeze(0)
  x_old = x
  x = x + signal
  return x

if __name__ == '__main__':
  ffn = FeedForwardNetwork(512, 2048, 0.1)
  x = torch.randn((5, 11, 512))
  x = ffn(x)
  print(x.shape)