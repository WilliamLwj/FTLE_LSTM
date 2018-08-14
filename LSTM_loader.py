import numpy as np
from scipy.io import loadmat
import torch
import torch.utils.data as data
import pdb
import pickle
import random
import torchvision.transforms as transforms
class FTLE(data.Dataset):
    def __init__(self, length, seq_length, img_transform=None, label_transform=None):
        self.path = '/home/liwj/FTLE/Data/data_set_{}_{}.mat'
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.length = length
        self.seq_length = seq_length
        image_normalization_mean = []
        image_normalization_std = []
        label_normalization_mean = []
        label_normalization_std = []
        for i in range(seq_length):
            image_normalization_mean.append(0)
            image_normalization_std.append(0.156)
            label_normalization_mean.append(0.019)
            label_normalization_std.append(0.0054)
        self.normalize = transforms.Normalize(mean=image_normalization_mean,
                                         std=image_normalization_std)

        self.label_normalize = transforms.Normalize(mean=label_normalization_mean,
                                               std=label_normalization_std)

    def __getitem__(self, index):
        flow_num = index // 8 + 1
        depth = index % 8 + 1
        self.path = self.path.format(flow_num, depth)

        self.data = loadmat(self.path)

        width = int(np.sqrt(self.data['U'].shape[0]))
        self.time_num = int(self.data['U'].shape[1])

        input_U= self.data['U'].reshape(width, width, self.time_num).transpose(2, 0, 1)
        input_V= self.data['V'].reshape(width, width, self.time_num).transpose(2, 0, 1)
        label = self.data['F_CMB'].reshape(width, width, self.time_num).transpose(2, 0, 1)
        input_U, input_V, label = self.shuffle(input_U, input_V, label)

        input_U, input_V, label = self.normalize(torch.from_numpy(input_U)), self.normalize(torch.from_numpy(input_V)),\
                                  self.label_normalize(torch.from_numpy(label))

        input = np.concatenate((input_U[:, np.newaxis, :, :], input_V[:, np.newaxis, :, :]), axis=1)
   

        if self.img_transform is not None:

            input = self.img_transform(input)
        if self.label_transform is not None:

            label = self.label_transform(label)

        return input, label
    def shuffle(self, input_U, input_V, label):
        p = random.randint(0, self.time_num)
        if p <= (self.time_num - self.seq_length):
            input_U = input_U[p:p + self.seq_length, :, :]
            input_V = input_V[p:p + self.seq_length, :, :]
            label = label[p:p + self.seq_length, :, :]
        else:
            input_U = np.concatenate((input_U[p:self.time_num, :, :], input_U[0:(self.seq_length - self.time_num + p), :, :]), axis=0)
            input_V = np.concatenate((input_V[p:self.time_num, :, :], input_V[0:(self.seq_length - self.time_num + p), :, :]), axis=0)
            label = np.concatenate((label[p:self.time_num, :, :], label[0: (self.seq_length - self.time_num + p), :, :]), axis=0)

        return input_U, input_V, label

    def __len__(self):
        return self.length



class FTLE_new(data.Dataset):
    def __init__(self, length, seq_length, train=True, img_transform=None, label_transform=None):
        self.path = '/home/liwj/FTLE/New_data/flowfield_1_{}_1.mat'
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.length = length
        self.seq_length = seq_length
        self.train = train
        image_normalization_mean = []
        image_normalization_std = []
        label_normalization_mean = []
        label_normalization_std = []
        for i in range(seq_length):
            image_normalization_mean.append(0)
            image_normalization_std.append(0.314)
            label_normalization_mean.append(0.234)
            label_normalization_std.append(0.090)
        self.normalize = transforms.Normalize(mean=image_normalization_mean,
                                              std=image_normalization_std)

        self.label_normalize = transforms.Normalize(mean=label_normalization_mean,
                                                    std=label_normalization_std)

    def __getitem__(self, index):
        if self.train:
            rem = 1
        else:
            rem = 21
        num = index % 100 + rem
        self.path = self.path.format(num)

        self.data = loadmat(self.path)

        width = int(np.sqrt(self.data['U'].shape[0]))
        self.time_num = int(self.data['U'].shape[1])

        input_U = self.data['U'].reshape(width, width, self.time_num).transpose(2, 0, 1)
        input_V = self.data['V'].reshape(width, width, self.time_num).transpose(2, 0, 1)
        label = self.data['FTLE_array'].reshape(width, width, self.time_num).transpose(2, 0, 1)

        input_U, input_V, label = self.shuffle(input_U, input_V, label)
        #input_U, input_V, label = self.normalize(torch.from_numpy(input_U)), self.normalize(torch.from_numpy(input_V)), \
        #                          self.label_normalize(torch.from_numpy(label))

        input = np.concatenate((input_U[:, np.newaxis, :, :], input_V[:, np.newaxis, :, :]), axis=1)

        if self.img_transform is not None:
            input = self.img_transform(input)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return input, label

    def shuffle(self, input_U, input_V, label):
        p = random.randint(0, self.time_num)
        if p <= (self.time_num - self.seq_length):
            input_U = input_U[p:p + self.seq_length, :, :]
            input_V = input_V[p:p + self.seq_length, :, :]
            label = label[p:p + self.seq_length, :, :]
        else:
            input_U = np.concatenate(
                (input_U[p:self.time_num, :, :], input_U[0:(self.seq_length - self.time_num + p), :, :]), axis=0)
            input_V = np.concatenate(
                (input_V[p:self.time_num, :, :], input_V[0:(self.seq_length - self.time_num + p), :, :]), axis=0)
            label = np.concatenate(
                (label[p:self.time_num, :, :], label[0: (self.seq_length - self.time_num + p), :, :]), axis=0)

        return input_U, input_V, label

    def __len__(self):
        return self.length