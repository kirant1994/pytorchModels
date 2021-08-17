import torch
from .attention import positional_encoding, prepare_additive_padding_mask, prepare_additive_lower_triangular_mask
from .encoder import Encoder
from .decoder import Decoder

class Transformer(torch.nn.Module):
  def __init__(self, input_dim, output_dim, args):
    super(Transformer, self).__init__()
    self.n_encoder = args.n_encoder
    self.n_decoder = args.n_decoder
    self.n_eheads = args.n_eheads
    self.n_dheads = args.n_dheads
    self.dropout_att = args.dropout_att
    self.dropout_relu = args.dropout_relu
    self.dropout_postprocess = args.dropout_postprocess
    self.n_efilters = args.n_efilters
    self.n_dfilters = args.n_dfilters
    self.input_dim = input_dim
    self.hidden_dim = args.hidden_dim
    self.output_dim = output_dim
    self.embedding_dim = args.embedding_dim

    self.fc_input = torch.nn.Linear(input_dim, self.hidden_dim, bias=True)
    self.target_space_embedding = torch.nn.Embedding(self.embedding_dim, self.hidden_dim)
    self.target_embedding = torch.nn.Embedding(self.output_dim, self.hidden_dim)
    self.dropout_encoder_preprocess = torch.nn.Dropout(1 - self.dropout_att)
    self.dropout_decoder_preprocess = torch.nn.Dropout(1 - self.dropout_att)

    self.encoder = Encoder(
                      n_layers=self.n_encoder,
                      hidden_dim=self.hidden_dim,
                      n_heads=self.n_eheads,
                      filter_size=self.n_efilters,
                      dropout_att_rate=self.dropout_att,
                      dropout_relu_rate=self.dropout_relu,
                      dropout_postprocess_rate=self.dropout_postprocess                
                    )
    self.decoder = Decoder(
                      n_layers=self.n_decoder,
                      hidden_dim=self.hidden_dim,
                      n_heads=self.n_dheads,
                      filter_size=self.n_dfilters,
                      dropout_att_rate=self.dropout_att,
                      dropout_relu_rate=self.dropout_relu,
                      dropout_postprocess_rate=self.dropout_postprocess                
                    )
    
    self.fc_output = torch.nn.Linear(self.hidden_dim, output_dim, bias=False)
  
  def forward(self, x, y, ilens, olens, target_space=3):
    # Fixing length for parallel processing
    max_ilen = torch.max(ilens)
    x = x[:, :max_ilen]
    target_space = torch.tensor(target_space, dtype=torch.int64).to(x.device)

    x, encoder_bias = self.prepare_inputs(x, ilens, target_space)
    z, decoder_bias = self.prepare_targets(y)

    x_enc, enc_att_list = self.encoder(x, encoder_bias)
    x_dec, (dec_self_att_list, dec_cross_att_list) = self.decoder(
                                                        z,
                                                        memory=x_enc,
                                                        self_bias=decoder_bias,
                                                        memory_bias=encoder_bias
                                                      )
    x = self.fc_output(x)
    x = torch.nn.functional.log_softmax(x, dim=-1)
    return x, (enc_att_list, dec_self_att_list, dec_cross_att_list)

  def prepare_inputs(self, x, ilens, target_space):
    encoder_bias = prepare_additive_padding_mask(ilens)
    x = self.fc_input(x)
    x_tse = self.target_space_embedding(target_space)
    x = x + x_tse
    x = positional_encoding(x)
    x = self.dropout_encoder_preprocess(x)
    return x, encoder_bias.unsqueeze(1) # Unsqueezing for the heads

  def prepare_targets(self, y):
    assert len(y.shape) == 2
    N, T = y.shape
    decoder_bias = prepare_additive_lower_triangular_mask(T, N)
    y = self.target_embedding(y)

    # Shifting to the right and adding a zero vector for SOS
    y = torch.nn.functional.pad(y, (0, 0, 1, 0))[:, :-1]
    y = positional_encoding(y)
    y = self.dropout_decoder_preprocess(y)
    return y, decoder_bias.unsqueeze(1) # Unsqueezing for the heads