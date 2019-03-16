from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import torchvision.models as models

import os
import numpy as np
import requests
import argparse
import logging
import time
import pickle
import copy

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from DRLoader import DSLoader

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 20)')
parser.add_argument('--batchSize', action='store', default=64, type=int, help='batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.0001, type=float, help='learning rate (default: 0.0001)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--preTrained_f', action='store_false', default=True, help='Flag to pretrained model (default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='AlexNet', const='AlexNet',nargs='?', choices=['AlexNet','VGG'], help="net model(default:AlexNet)")
arg = parser.parse_args()

def main():
    batchSize = arg.batchSize
    # create model directory to store/load old model
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
        
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_SingleImgs_'+arg.net+'_'+str(arg.lr)+'.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Classifier: "+arg.net)
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    # Batch size setting
    batchSize = arg.batchSize
    
    # load the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    torch.cuda.set_device(arg.gpu_num)
    torch.cuda.current_device()
    
    pickle_file = 'UCF_imgs.pickle'
    num_classes=11
    data=pickle.load(open(pickle_file,'rb'))
    train_data=data['train']
    test_data=data['test']
    
    train_dataset = DSLoader(train_data,data_transforms['train'])
    test_dataset = DSLoader(test_data,data_transforms['test'])
    train_len = train_dataset.__len__()
    test_len = test_dataset.__len__()
    
    trainLoader = DataLoader(train_dataset,batch_size=batchSize, shuffle=True, num_workers=4)
    testLoader = DataLoader(test_dataset,batch_size=batchSize, shuffle=False, num_workers=4)
    
    if arg.net == 'VGG':
        model = models.vgg16(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096,num_classes)
    elif arg.net == 'AlexNet':
        model = models.alexnet(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096,num_classes)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    
    # for gpu mode
    if arg.useGPU_f:
        model.cuda()
    
    model_path = 'model/model_SingleImg_'+arg.net+'_'+str(arg.lr)+'.pt'
            
    ### Training ###
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs if arg.train_f else 0
    
    min_accuracy = 0
    correct_label , ave_loss = 0, 0
    
    for epoch in xrange(epochs):
        # training
        for par in model.parameters():
            par.requires_grad=True
        model.train()
        
        for batchIndex,(imageBatch,imageID) in enumerate(trainLoader):
            optimizer.zero_grad()
            # for gpu mode
            if arg.useGPU_f:
                input, target = Variable(imageBatch.cuda(),requires_grad=True), Variable(imageID.cuda())
            # for cpu mode
            else:
                input, target = Variable(imageBatch,requires_grad=True), Variable(imageID)

            # use cross entropy loss
            criterion = nn.CrossEntropyLoss()
            classEstimates = model(input)
            loss = criterion(classEstimates, target)
            _, pred_label = torch.max(classEstimates.data, 1)
            correct_label = (pred_label == target.data).sum()
            label_accuracy = correct_label.data.cpu().numpy()*100.0/batchSize

            loss.backward()              
            optimizer.step()      
            
            if batchIndex%10==0:
                logger.info('==>>> epoch:{}, batch index: {}, train loss:{}, label accuracy:{}'.format(epoch,batchIndex, loss.data.cpu().numpy(), label_accuracy))
                
        ##### START VALIDATION #####
        overall_acc, overall_loss = 0.0, 0.0
        torch.no_grad()
        model.eval()
        for batchIndex,(imageBatch,imageID) in enumerate(testLoader):
            optimizer.zero_grad()
            # for gpu mode
            if arg.useGPU_f:
                input, target = Variable(imageBatch.cuda(),requires_grad=False), Variable(imageID.cuda())
            # for cpu mode
            else:
                input, target = Variable(imageBatch,requires_grad=False), Variable(imageID)

            # use cross entropy loss
            criterion = nn.CrossEntropyLoss()
            classEstimates = model(input)
            loss = criterion(classEstimates, target)
            _, pred_label = torch.max(classEstimates.data, 1)
            correct_label = (pred_label == target.data).sum()
            overall_acc += correct_label
            overall_loss += loss.data.cpu().numpy()
        val_accuracy = overall_acc.data.cpu().numpy()*100.0/test_len
        val_loss = overall_loss*100.0/test_len
        
        if val_accuracy > min_accuracy:
            min_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
        
        logger.info('==>>> Val loss:{}, accuracy:{}'.format(val_loss, val_accuracy))
            
    ##### START TESTING #####
    print("Start Testing for Best Model")
    logger.info("Start Testing for Best Model")
    if os.path.isfile(model_path):
        print("Loading Model ...")
        model.load_state_dict(torch.load(model_path))
    
    #switch to evaluate mode
    torch.no_grad()
    model.eval()
    
    overall_acc, overall_loss = 0.0, 0.0
    for batchIndex,(imageBatch,imageID) in enumerate(testLoader):
        optimizer.zero_grad()
        # for gpu mode
        if arg.useGPU_f:
            input, target = Variable(imageBatch.cuda(),requires_grad=True), Variable(imageID.cuda())
        # for cpu mode
        else:
            input, target = Variable(imageBatch,requires_grad=True), Variable(imageID)

        # use cross entropy loss
        criterion = nn.CrossEntropyLoss()
        classEstimates = model(input)
        loss = criterion(classEstimates, target)
        _, pred_label = torch.max(classEstimates.data, 1)
        correct_label = (pred_label == target.data).sum()
        overall_acc += correct_label
        overall_loss += loss.data.cpu().numpy()
    test_accuracy = overall_acc.data.cpu().numpy()*100.0/test_len
    test_loss = overall_loss*100.0/test_len
    logger.info('==>>> Val loss:{}, accuracy:{}'.format(test_loss, test_accuracy))
    
if __name__ == "__main__":
    main()