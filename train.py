import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable, grad
from torch.optim import Adam
import torchvision.transforms as transforms
from models import *
from LSTM_loader import *

import argparse

# parse for config file
parser = argparse.ArgumentParser()
parser.add_argument('-c', "--config", default='config', type=str,)
args = parser.parse_args()
config_file = args.config
module = __import__(config_file)
params = module.params

ftle = FTLE(100, 21)
ftle_new = FTLE_new(100, 21)
loader1 = data.DataLoader(ftle, batch_size=params['batch_size'])
loader2 = data.DataLoader(ftle_new, batch_size=params['batch_size'])
model = TimeMachine(params).cuda(params['device'])
model.train()
criterion = nn.MSELoss().cuda(params['device'])

optimizer = Adam(model.parameters(), lr=params['lr'])

loader = loader2
for e in range(params['num_epoch']):

	for idx, (seqs, labs) in enumerate(loader):
		seqs, labs = (Variable(seqs).cuda(params['device'])).float(), (Variable(labs).cuda(params['device'])).float()
		N_1, N_2, C, H, W = seqs.size()
		preds = model(seqs)

		valid_preds, valid_labs = preds[:, params['warm_up']:,], labs[:, :-params['warm_up'], ]

		valid_preds, valid_labs = valid_preds.contiguous().view(-1, 1, H, W), valid_labs.contiguous().view(-1, 1, H, W)

		loss = criterion(valid_preds, valid_labs)
		print('Loss:   ', loss.data.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

torch.save(model.state_dict(), '/home/liwj/FTLE/LSTM/LSTM_combine.pth')







