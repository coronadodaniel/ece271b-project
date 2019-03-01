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
    if not os.path.exists('model'):
        os.makedirs('model')
    model_path = 'model/model_dLSTM.pt'
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_dLSTM.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    logger.info("Batch Size: {}".format(arg.batchSize))
    logger.info("Window Size: {}".format(arg.windowSize))
    logger.info("Hidden Layer Dimension: {}".format(arg.h_dim))
    
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
    trainSize = trainLoader.__len__()
    testSize = testLoader.__len()
    
    model = dVGG(arg.h_dim, num_of_classes)
    
    if arg.useGPU_f:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    
    if arg.useGPU_f:
        s=Variable(torch.zeros(arg.batchSize,arg.windowSize).cuda(),requires_grad=False)
        h=Variable(torch.zeros(arg.batchSize,arg.windowSize).cuda(),requires_grad=False)
        dv=Variable(torch.zeros(arg.batchSize,arg.windowSize).cuda(),requires_grad=False)
    else:
        s=Variable(torch.zeros(arg.batchSize,arg.windowSize),requires_grad=False)
        h=Variable(torch.zeros(arg.batchSize,arg.windowSize),requires_grad=False)
        dv=Variable(torch.zeros(arg.batchSize,arg.windowSize),requires_grad=False)
     
    min_acc=0.0
    ##########################
    ##### Start Training #####
    ##########################
    epochs = arg.epochs if arg.train_f else 0
    for epoch in range(epochs):
        for batchIdx,(windowBatch,labelBatch) in enumerate(trainLoader(arg.batchSize)):
            y=torch.zeros
            for i in range(arg.windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                if arg.useGPU_f:
                    imgBatch = Variable(imgBatch.cuda(),requires_grad=True)
                    labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
                else:
                    imgBatch = Variable(imgBatch,requires_grad=True)
                    labelBatch = Variable(labelBatch,requires_grad=False)


                y,h,dv,s = model(windowBatch,h,dv,s) ## need to append y
                loss = criterion(y,labelBatch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                _,pred = torch.max(y,1) ### prediction should after averging the array
                train_acc = (pred == labelBatch.data).sum()
                train_acc = train_acc/batchSize

            if batchIdx%100==0:
                logger.info("epochs:{}, train loss:{}, train acc:{}".format(epoch, loss.data.cpu(), train_acc))
        
        ########################
        ### Start Validation ###
        ########################
        val_acc=0.0
        for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader(arg.batchSize)):
            if arg.useGPU_f:
                windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
            else:
                windowBatch = Variable(windowBatch,requires_grad=True)
                labelBatch = Variable(labelBatch,requires_grad=False)

            y,h,dv,s = model(windowBatch,h,dv,s)
            loss = criterion(y,labelBatch)

            _,pred = torch.max(y,1)
            val_acc += (pred == labelBatch.data).sum()
        val_acc = train_acc/testSize
        logger.info("==> val loss:{}, val acc:{}".format(val_acc,loss))
        
        if val_acc>min_cc:
            min_acc=val_acc
            torch.save(model.state_dict(), model_path)
            
    ##########################
    ##### Start Testing #####
    ##########################        
    test_acc=0.0
        for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader(arg.batchSize)):
            if arg.useGPU_f:
                windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
            else:
                windowBatch = Variable(windowBatch,requires_grad=True)
                labelBatch = Variable(labelBatch,requires_grad=False)

            y,h,dv,s = model(windowBatch,h,dv,s)
            loss = criterion(y,labelBatch)

            _,pred = torch.max(y,1)
            test_acc += (pred == labelBatch.data).sum()
        test_acc = train_acc/testSize
            

if __name__ == "__main__":
    main()