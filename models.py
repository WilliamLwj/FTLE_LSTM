from torch import nn
from torch.nn import functional
from torch.autograd import Variable, grad

from conv_lstm import ConvLSTM
import pdb

def build_conv_block(down, in_channel, out_channel, args, activation='none', batch_norm=True):
	modules = []
	if down:
		modules.append(nn.Conv2d(in_channel, out_channel, *args, bias = not batch_norm))
	else:
		modules.append(nn.ConvTranspose2d(in_channel, out_channel, *args, bias = not batch_norm))

	if batch_norm:
		modules.append(nn.BatchNorm2d(out_channel))
	if activation == 'relu':
		modules.append(nn.ReLU())
	elif activation == 'tanh':
		modules.append(nn.Tanh())

	return nn.Sequential(*modules)

def downsize(size, to_go):
	return size if to_go <= 0 else downsize(int(size/2)+size%2, to_go-1)



class TimeMachine(nn.Module):

	def __init__(self, params):
		super(TimeMachine, self).__init__()
		# build encoder decoder
		self.encoder = nn.Sequential(*[build_conv_block(*config) for config in params['encoder']])
		self.decoder = nn.Sequential(*[build_conv_block(*config) for config in params['decoder']])
		#self.decoder = build_conv_block(*(params['decoder'][0]))
		# build lstm
		_, height, width = params['input_dim']
		latent_dim = params['encoder'][-1][2]*downsize(height, len(params['encoder']))*downsize(width, len(params['encoder']))
		self.lstm = nn.LSTM(latent_dim, latent_dim, 1, bias=True, batch_first=params['batch_first'])

	def forward(self, inputs):
		N_1, N_2, _, H, W = inputs.size()
		# Pass encoder
		features = self.encoder(inputs.view(-1, 2, H, W))
		_, C, fH, fW = features.size()
		features = features.view(N_1, N_2, -1).contiguous()
		latent_codes, (h_n, c_n) = self.lstm(features.permute(1, 0, 2))
		
		latent_codes = latent_codes.view(N_2, N_1, C, fH, fW).contiguous().view(-1, C, fH, fW)
		output_field = self.decoder(latent_codes)

		return output_field.view(N_2, N_1, 1, H, W).permute(1, 0, 2, 3, 4)
















