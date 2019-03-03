import torch
import os
import sys
import pickle
import random
import argparse
import logging
import imageio

import torchvision.models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import nn

import matplotlib.pyplot as plt

from model import dVGG
from DRLoader import DRLoader

plt.switch_backend('agg')

imageio.plugins.ffmpeg.download()

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 20)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--windowSize', action='store', default=25, type=int, help='number of frames (default: 25)')
parser.add_argument('--h_dim', action='store', default=256, type=int, help='LSTM hidden layer dimension (default: 256)')
parser.add_argument('--lr','--learning-rate',action='store',default=0.01, type=float,help='learning rate (default: 0.01)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
arg = parser.parse_args()

def main():
    torch.cuda.set_device(arg.gpu_num)
    torch.cuda.current_device()

    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
    model_path = 'model/model_dLSTM.pt'

    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    if not os.path.exists('log'):
        os.makedirs('log')
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

    trainLoader = DRLoader(train_path, arg.windowSize, data_transforms['train'], True)
    testLoader = DRLoader(test_path, arg.windowSize, data_transforms['test'], False)
    trainSize = trainLoader.__len__()
    testSize = testLoader.__len__()

    model = dVGG(arg.h_dim, num_of_classes)

    if arg.useGPU_f:
        model.cuda()

    optimizer = optim.Adam(model.parameters(),lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    if arg.useGPU_f:
        s=Variable(torch.zeros(arg.batchSize,arg.h_dim).cuda(),requires_grad=False)
        h=Variable(torch.zeros(arg.batchSize,arg.h_dim).cuda(),requires_grad=False)
        dv=Variable(torch.zeros(arg.batchSize,arg.h_dim).cuda(),requires_grad=False)
    else:
        s=Variable(torch.zeros(arg.batchSize,arg.h_dim),requires_grad=False)
        h=Variable(torch.zeros(arg.batchSize,arg.h_dim),requires_grad=False)
        dv=Variable(torch.zeros(arg.batchSize,arg.h_dim),requires_grad=False)

    train_acc_plot = []
    train_loss_plot = []
    test_acc_plot = []
    test_loss_plot = []

    min_acc=0.0
    ##########################
    ##### Start Training #####
    ##########################
    #torch.no_grad() #####
    training_iters = len(trainLoader)//arg.batchSize
    test_iters = len(testLoader)//arg.batchSize
    epochs = arg.epochs if arg.train_f else 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_size = 0
        train_loss = 0
        train_acc = 0
        for batchIdx,(windowBatch,labelBatch) in enumerate(trainLoader.batches(arg.batchSize)):
            loss=0.0
            y=torch.zeros(arg.batchSize, num_of_classes).cuda()
            if arg.useGPU_f:
                windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
            else:
                windowBatch = Variable(windowBatch,requires_grad=True)
                labelBatch = Variable(labelBatch,requires_grad=False)

            for i in range(arg.windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                temp,h,dv,s = model(imgBatch,h,dv,s)
                h,dv,s = h.detach(), dv.detach(), s.detach()
                #loss_ = criterion(temp,labelBatch)
                #loss+=loss_.data
                y += temp

            Y=y/arg.windowSize
            #loss = Variable(loss.cuda(),requires_grad=True)
            loss = criterion(Y, labelBatch)
            train_loss += loss.item()
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _,pred = torch.max(Y,1) ### prediction should after averging the array
            train_acc += (pred == labelBatch.data).sum()
            #train_acc = train_acc.data.cpu().numpy()/arg.batchSize
            train_size += arg.batchSize


            if batchIdx%100==0:
                logger.info("epochs:{}, iteration:{}/{}, train loss:{}".format(epoch, batchIdx, training_iters, loss.data.cpu()))

        train_loss = train_loss/train_size
        train_acc =  train_acc.data.cpu().numpy()/train_size
        print("train_acc = ",train_acc)
        train_loss_plot.append(train_loss)
        train_acc_plot.append(train_acc)


        print(train_acc_plot)


        ########################
        ### Start Validation ###
        ########################
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        val_size = 0
        for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader.batches(arg.batchSize)):
            y=torch.zeros(arg.batchSize, num_of_classes).cuda()
            if arg.useGPU_f:
                windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
            else:
                windowBatch = Variable(windowBatch,requires_grad=True)
                labelBatch = Variable(labelBatch,requires_grad=False)
            for i in range(arg.windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                temp,h,dv,s = model(imgBatch,h,dv,s)
                h,dv,s = h.detach(), dv.detach(), s.detach()
                #loss_ = criterion(temp,labelBatch)
                #loss+=loss_.data
                y += temp

            Y=y/arg.windowSize
            loss = criterion(Y,labelBatch)
            val_loss += loss.item()
            _,pred = torch.max(Y,1)
            val_acc += (pred == labelBatch.data).sum()
            #v_acc = v_acc.data.cpu().numpy()/arg.batchSize
            val_size += arg.batchSize

        val_loss = val_loss/val_size
        val_acc = val_acc.data.cpu().numpy()/val_size
        logger.info("==> val loss:{}, val acc:{}".format(val_loss, val_acc))

        test_loss_plot.append(val_loss)
        test_acc_plot.append(val_acc)

        if val_acc>min_acc:
            min_acc=val_acc
            torch.save(model.state_dict(), model_path)

    ##########################
    ##### Start Testing #####
    ##########################
    model.eval()
    torch.no_grad()
    test_acc=0.0
    test_loss=0.0
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))

    test_size = 0
    for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader.batches(arg.batchSize)):
        y=torch.zeros(arg.batchSize, num_of_classes).cuda()
        if arg.useGPU_f:
            windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
            labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
        else:
            windowBatch = Variable(windowBatch,requires_grad=True)
            labelBatch = Variable(labelBatch,requires_grad=False)


        for i in range(arg.windowSize):
            imgBatch = windowBatch[:,i,:,:,:]
            temp,h,dv,s = model(imgBatch,h,dv,s)
            h,dv,s = h.detach(), dv.detach(), s.detach()
            #loss_ = criterion(temp,labelBatch)
            #loss+=loss_.data
            y += temp

        Y=y/arg.windowSize
        loss = criterion(Y,labelBatch)
        test_loss += loss.item()
        _,pred = torch.max(Y,1)
        test_acc += (pred == labelBatch.data).sum()
        #t_acc = t_acc.data.cpu().numpy()/arg.batchSize
        test_size += arg.batchSize

    test_loss = test_loss/ test_size
    test_acc =  test_acc.data.cpu().numpy()/test_size
    logger.info("==> test loss:{}, test acc:{}".format(test_loss,test_acc))



    x = list(range(len(test_loss_plot)))
    print(x)
    plt.figure()
    plt.plot(x, train_loss_plot, label="train loss")
    plt.plot(x, test_loss_plot, label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("dLSTM/dVGG Loss")
    plt.savefig('dLSTM_loss.png')

    x = list(range(len(test_acc_plot)))
    plt.figure()
    plt.plot(x, train_acc_plot, label="train accuracy")
    plt.plot(x, test_acc_plot, label="test accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("dLSTM/dVGG Accuracy")
    plt.savefig('dLSTM_accuracy.png')



if __name__ == "__main__":
    main()
