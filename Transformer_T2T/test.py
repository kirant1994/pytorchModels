from transformer import transformer, hparams
from transformer.encoder import Encoder
from transformer.decoder import DecoderLayer
import torch

args = hparams.Hparams()
model = transformer.Transformer(input_dim=240, output_dim=1088, args=args).double()

# print(model)
X = torch.randn((5, 11, 240)).double()
y = torch.randint(0, 1024, (5, 7)).long()
ilens = torch.randint(1, 11, (5,)).long()
olens = torch.randint(1, 7, (5,)).long()
# y_pred, attentions = model(X, y, ilens, olens)

# print(y_pred.shape)
mdict = model.state_dict()

for key in mdict.keys():
  print(key, mdict[key].shape)