import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, dim, signal_period=500, trainable=False):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.dim = dim
        self._dim = int(math.ceil(float(self.dim) / 2) * 2)
        # The period after which the sine waves sould repeat in dimension
        self.wavelength_scale = 10
        self.signal_period = signal_period

        self.drop = torch.nn.Dropout(0.1)

        # Creating blank result
        self.grid = torch.zeros((1, self.max_len, self.dim), dtype=torch.float)
        # Creating the axes
        t_axis = torch.arange(0, self.max_len, dtype=torch.float) * math.pi * 2 / self.signal_period
        d_axis = torch.arange(0, self._dim, 2, dtype=torch.float)
        numerator = t_axis.view((-1, 1))
        denominator_inv = torch.exp(d_axis * -math.log(self.wavelength_scale) * 2 / self.dim)
        product = numerator * denominator_inv

        self.grid[0, :, 0::2] = torch.sin(product)
        self.grid[0, :, 1::2] = torch.cos(product)

        # self.grid = torch.nn.Parameter(self.grid)
        # self.register_parameter('grid', self.grid)
        # self.grid.requires_grad = trainable
    
    # Expects inputs of the shape NxTxD
    def forward(self, x):

        # x = self.drop(x)
        assert x.size(1) <= self.max_len, 'Maximum length defined for the positional encoder is {0:d}. Got {1:d}'.format(self.max_len, x.size(1))
        x = x + self.grid[:, :x.size(1)].to(x.device)
        return x
