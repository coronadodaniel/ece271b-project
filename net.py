import torch
import os
import sys
import pickle
import random
import argparse
import logging

import torchvision.models as models
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model import dVGG
from DRLoader import DRLoader

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 20)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--windowSize', action='store', default=25, type=int, help='number of frames (default: 25)')
parser.add_argument('--h_dim', action='store', default=256, type=int, help='LSTM hidden layer dimension (default: 256)')
parser.add_argument('--lr','--learning-rate',action='store',default=0.01, type=float,help='learning rate (default: 0.01)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
arg = parser.parse_args()

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.6,1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    root_dir = 'UCF11_split'
    train_path = root_dir+'/train'
    test_path = root_dir+'/test'
    num_of_classes=11
    
    trainLoader = DRLoader(train_path, arg.windowSize, data_transforms['train'])
    testLoader = DRLoader(test_path, arg.windowSize, data_transforms['test'])
    
    model = dVGG(arg.h_dim, num_of_classes)
    
    if arg.useGPU_f:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    
    ##########################
    ##### Start Training #####
    ##########################
    for windowBatch, labelBatch in trainLoader(arg.batchSize):
        if arg.useGPU_f:
            windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
            labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
        else:
            windowBatch = Variable(windowBatch,requires_grad=True)
            labelBatch = Variable(labelBatch,requires_grad=False)


if __name__ == "__main__":
    main()