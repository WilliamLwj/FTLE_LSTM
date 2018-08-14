from matplotlib import pyplot as plt
plt.switch_backend('agg')
import torch
from torch.utils import data

from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms

import pdb
from LSTM_loader import FTLE, FTLE_new
from models import TimeMachine
import numpy as np



model_name = 'LSTM_combine2'
model_path = '/home/liwj/FTLE/LSTM/LSTM_combine.pth'

def visualize_flow_by_quiver(ax, flow, density=(1.0/10)):
    flow = np.squeeze(flow).transpose(1,2,0)
    H, W, _  = flow.shape
    stride = int(1.0/density)
    Y, X = np.mgrid[0:H:stride, 0:W:stride]
    U, V = flow[::stride,::stride,0], flow[::stride,::stride,1]
    ax.imshow(np.zeros((H, W)))
    ax.quiver(X, Y, U, V, angles='xy', scale=5, color='r')
    #ax.axis('off')
    ax.set_title('Velocities')

def visualize_FTLE_with_color(fig, ax, FTLE, vmin, vmax, GT=True):

    im = ax.imshow(FTLE, vmin=vmin, vmax=vmax)

    #ax.axis('off')
    if GT:

        ax.set_title('FTLE Values (%.2f, %.2f)'%(vmin, vmax))
    else:
        ax.set_title('Predicted (%.2f, %.2f)' %(FTLE.min(), FTLE.max()))
    return im

def visualize_fluid():
    # load model
    from config import params
    model = TimeMachine(params)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.cuda()

    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,wspace=0.2,hspace=0.1)

    #train_set = FTLE(10, 21)
    train_set = FTLE_new(100, 21)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=True)
    train_iter = iter(train_loader)
    criterion = nn.MSELoss()

    velocity, GT = train_iter.next()

    velocity = Variable(velocity.float()).cuda()
    GT = Variable(GT.float()).cuda()
    pred = model(velocity)
    _, _, _, H, W = velocity.size()
    valid_preds, valid_labs = pred[:, params['warm_up']:, ], GT[:, :-params['warm_up'], ]
    valid_preds, valid_labs = valid_preds.contiguous().view(-1, 1, H, W), valid_labs.contiguous().view(-1, 1, H, W)

    loss = criterion(valid_preds, valid_labs)
    print('Relative loss: ', (np.sqrt(loss.data.item()) / (GT.norm())).item())  # 1.7784939700504765e-05
    velocity = velocity.cpu().data.numpy()
    pred = valid_preds.cpu().data.numpy()
    GT = valid_labs.cpu().data.numpy()
    for i in range(GT.shape[0]):
        vmin = GT[i][0].min()
        vmax = GT[i][0].max()
        visualize_flow_by_quiver(axes[0], velocity[0][i])
        im = visualize_FTLE_with_color(fig, axes[1], GT[i][0], vmin=vmin, vmax=vmax, GT=True)

        im = visualize_FTLE_with_color(fig, axes[2], pred[i][0], vmin=vmin, vmax=vmax, GT=False)

        fig.savefig('/home/liwj/FTLE/LSTM_result/' + model_name+'_{}.png'.format(i), dpi=300)
        plt.close()

visualize_fluid()














