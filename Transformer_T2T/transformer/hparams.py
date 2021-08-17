class Namespace:
  def __init__(self, **kwargs):
    self.__dict__.update(**kwargs)
  
  def __str__(self):
    text = ''
    for key in self.__dict__:
      if not key.startswith('_'):
        text += '{0:s}={1}\n'.format(key, self.__dict__[key])
    return text.strip()

class Hparams(Namespace):
  def __init__(
    self,
    n_encoder=6,
    n_eheads=16,
    n_efilters=3072,
    n_decoder=4,
    n_dheads=4,
    n_dfilters=3072,
    hidden_dim=512,
    dropout_att=0.1,
    dropout_relu=0.1,
    dropout_postprocess=0.2,
    embedding_dim=32
  ):
    super(Hparams, self).__init__(
      n_encoder=n_encoder,
      n_eheads=n_eheads,
      n_efilters=n_efilters,
      n_decoder=n_decoder,
      n_dheads=n_dheads,
      n_dfilters=n_dfilters,
      hidden_dim=hidden_dim,
      dropout_att=dropout_att,
      dropout_relu=dropout_relu,
      dropout_postprocess=dropout_postprocess,
      embedding_dim=embedding_dim,
    )

if __name__ == '__main__':
  hparams = Hparams()
  print(hparams)