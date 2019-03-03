import torch
import os
import sys
import pickle
import random
import argparse
import logging

import torchvision.models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model import dVGG, dAlexNet
from DRLoader import DRLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 20)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--windowSize', action='store', default=25, type=int, help='number of frames (default: 25)')
parser.add_argument('--h_dim', action='store', default=256, type=int, help='LSTM hidden layer dimension (default: 256)')
parser.add_argument('--lr','--learning-rate',action='store',default=0.01, type=float,help='learning rate (default: 0.01)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num','--list', type=int, nargs='+',help='gpu_num ',required=True)
parser.add_argument("--net", default='dAlexNet', const='dAlexNet',nargs='?', choices=['dVGG', 'dAlexNet'], help="net model(default:VGG)")

arg = parser.parse_args()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
def main():
    if len(arg.gpu_num)==1:
        torch.cuda.set_device(arg.gpu_num[0])
    
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
    model_path = 'model/model_dLSTM_'+str(arg.lr)+'_'+arg.net+'.pt'
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_dLSTM_'+str(arg.lr)+'_'+arg.net+'.log')
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
    logger.info("GPU num: {}".format(arg.gpu_num))
    
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
    testLoader = DRLoader(test_path, arg.windowSize, data_transforms['test'], True)
    trainSize = trainLoader.__len__()
    testSize = testLoader.__len__()
    
    if arg.net == 'dVGG':
        model = dVGG(arg.h_dim, num_of_classes)
    elif arg.net == 'dAlexNet':
        model = dAlexNet(arg.h_dim, num_of_classes)
    
    if arg.useGPU_f:
        if len(arg.gpu_num)>1:
            model = nn.DataParallel(model,arg.gpu_num,dim=0)
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    
    if arg.useGPU_f:
        s=Variable(torch.randn(arg.batchSize,arg.h_dim).cuda(),requires_grad=False)
        h=Variable(torch.randn(arg.batchSize,arg.h_dim).cuda(),requires_grad=False)
        dv=Variable(torch.randn(arg.batchSize,arg.h_dim).cuda(),requires_grad=False)
        (h0,c) = ( Variable(torch.randn(arg.batchSize,arg.h_dim).cuda(),requires_grad=False),
                   Variable(torch.randn(arg.batchSize,arg.h_dim).cuda(),requires_grad=False))
    else:
        s=Variable(torch.randn(arg.batchSize,arg.h_dim),requires_grad=False)
        h=Variable(torch.randn(arg.batchSize,arg.h_dim),requires_grad=False)
        dv=Variable(torch.randn(arg.batchSize,arg.h_dim),requires_grad=False)
        (h0,c) = ( Variable(torch.randn(arg.batchSize,arg.h_dim),requires_grad=False),
                   Variable(torch.randn(arg.batchSize,arg.h_dim),requires_grad=False))
     
    min_acc=0.0
    ##########################
    ##### Start Training #####
    ##########################
    epochs = arg.epochs if arg.train_f else 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for batchIdx,(windowBatch,labelBatch) in enumerate(trainLoader.batches(arg.batchSize)):
            #loss=0.0
            if arg.useGPU_f:
                y=torch.zeros(arg.batchSize, num_of_classes).cuda()
                windowBatch = Variable(windowBatch.cuda(),requires_grad=True)
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
            else:
                y=torch.zeros(arg.batchSize, num_of_classes)
                windowBatch = Variable(windowBatch,requires_grad=True)
                labelBatch = Variable(labelBatch,requires_grad=False)
            
            for i in range(arg.windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                temp,(h0,c),h,dv,s = model(imgBatch,(h0,c),h,dv,s)
                #(h0,c) = hidden
                (h0,c) = (h0.detach(), c.detach())
                h,dv,s = h.detach(), dv.detach(), s.detach()
                #loss_ = criterion(temp,labelBatch)
                #loss+=loss_.data
                y += temp
            
            Y=y/arg.windowSize
            #loss = Variable(loss.cuda(),requires_grad=True)
            loss = criterion(Y,labelBatch)
            loss.backward()
            #plot_grad_flow(model.named_parameters())
            optimizer.step()
            optimizer.zero_grad()

            _,pred = torch.max(Y,1) ### prediction should after averging the array
            train_acc = (pred == labelBatch.data).sum()
            train_acc = 100.0*train_acc.data.cpu().numpy()/arg.batchSize
            #print('train acc', train_acc, 'train loss', loss.data.cpu())

            if batchIdx%100==0:
                logger.info("epochs:{}, batchIdx:{}, train loss:{}, train acc:{}".format(epoch, batchIdx, loss.data.cpu(), train_acc))
        
        ########################
        ### Start Validation ###
        ########################
        model.eval()
        val_acc=0.0
        for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader.batches(arg.batchSize)):
            if arg.useGPU_f:
                y=torch.zeros(arg.batchSize, num_of_classes).cuda()
                windowBatch = Variable(windowBatch.cuda(),requires_grad=False)
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
            else:
                y=torch.zeros(arg.batchSize, num_of_classes)
                windowBatch = Variable(windowBatch,requires_grad=False)
                labelBatch = Variable(labelBatch,requires_grad=False)
            for i in range(arg.windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                temp,(h0,c),h,dv,s = model(imgBatch,(h0,c),h,dv,s)
                #(h0,c) = hidden
                (h0,c) = (h0.detach(), c.detach())
                h,dv,s = h.detach(), dv.detach(), s.detach()
                #loss_ = criterion(temp,labelBatch)
                #loss+=loss_.data
                y += temp
            
            Y=y/arg.windowSize
            loss = criterion(Y,labelBatch)

            _,pred = torch.max(Y,1)
            val_acc += (pred == labelBatch.data).sum()
            
        val_acc = 100.0*val_acc.data.cpu().numpy()/testSize
        logger.info("==> val loss:{}, val acc:{}".format(val_acc,loss.data.cpu().numpy()))
        
        if val_acc>min_acc:
            min_acc=val_acc
            torch.save(model.state_dict(), model_path)
    #plt.show()        
    ##########################
    ##### Start Testing #####
    ##########################   
    model.eval()
    torch.no_grad()
    test_acc=0.0
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        
    for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader.batches(arg.batchSize)):
        y=torch.zeros(arg.batchSize, num_of_classes)
        if arg.useGPU_f:
            y=torch.zeros(arg.batchSize, num_of_classes).cuda()
            windowBatch = Variable(windowBatch.cuda(),requires_grad=False)
            labelBatch = Variable(labelBatch.cuda(),requires_grad=False)
        else:
            y=torch.zeros(arg.batchSize, num_of_classes)
            windowBatch = Variable(windowBatch,requires_grad=False)
            labelBatch = Variable(labelBatch,requires_grad=False)
        
        for i in range(arg.windowSize):
            imgBatch = windowBatch[:,i,:,:,:]
            temp,(h0,c),h,dv,s = model(imgBatch,(h0,c),h,dv,s)
            #(h0,c) = hidden
            (h0,c) = (h0.detach(), c.detach())
            h,dv,s = h.detach(), dv.detach(), s.detach()
            #loss_ = criterion(temp,labelBatch)
            #loss+=loss_.data
            y += temp

        Y=y/arg.windowSize
        loss = criterion(Y,labelBatch)
        _,pred = torch.max(y,1)
        test_acc += (pred == labelBatch.data).sum()
    test_acc = 100.0*train_acc.data.cpu().numpy()/testSize
    
    logger.info("==> test loss:{}, test acc:{}".format(test_acc,loss.data.cpu().numpy()))


if __name__ == "__main__":
    main()
