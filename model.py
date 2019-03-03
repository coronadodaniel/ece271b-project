import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import torchvision.models as models
import copy

class dLSTM(nn.Module):
    def __init__(self, h_dim):
        super(dLSTM, self).__init__()
        
        in_dim = h_dim
        cat_dim = in_dim+(2*h_dim)        
        self.i = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.f = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.o = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.g = nn.Sequential(nn.Linear(in_dim+h_dim,h_dim), nn.Tanh())
        self.h_dim = h_dim
        
    def forward(self,x,h,dv,s):
        x=x.view(-1,self.h_dim)
        i = self.i(torch.cat([x,h,dv],1))
        f = self.f(torch.cat([x,h,dv],1))
        g = self.g(torch.cat([x,h],1))
        o = self.o(torch.cat([x,h,dv],1))
        
        s_ = torch.add(torch.mul(f,s),torch.mul(i,g))
        h = torch.mul(o,torch.tanh(s_))
        dv = s_-s
        return h, dv, s_
    
class d2LSTM(nn.Module):
    def __init__(self, h_dim):
        super(d2LSTM, self).__init__()
        
        in_dim = h_dim
        cat_dim = in_dim+(2*h_dim)        
        self.i = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.f = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.o = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.g = nn.Sequential(nn.Linear(in_dim+h_dim,h_dim), nn.Tanh())
        self.h_dim = h_dim
        
    def forward(self,x,h,dv2,s,s__):
        x=x.view(-1,self.h_dim)
        i = self.i(torch.cat([x,h,dv2],1))
        f = self.f(torch.cat([x,h,dv2],1))
        g = self.g(torch.cat([x,h],1))
        o = self.o(torch.cat([x,h,dv2],1))
        
        s_ = torch.add(torch.mul(f,s),torch.mul(i,g))
        h = torch.mul(o,torch.tanh(s_))
        dv2 = s_-(2*s)+s__
        return h, dv2,s, s_
    
class dVGG(nn.Module):
    def __init__(self, h_dim,num_of_classes):
        super(dVGG, self).__init__()
        
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
        self.classifier = nn.Linear(h_dim,num_of_classes)
        self.LSTM = nn.LSTM(4096,h_dim)
        self.dLSTM = dLSTM(h_dim)
        self.d2LSTM = d2LSTM(h_dim)
        self.h_dim=h_dim
        
    def forward(self,x,(h0,c),h1,dv1,s,h2,dv2,s_):
        f = self.model(x)
        f = f.view(1,-1,4096)
        hidden = (h0.view(1,-1,self.h_dim),c.view(1,-1,self.h_dim))
        out,hidden = self.LSTM(f,hidden)
        h1,dv1,s = self.dLSTM(out,h1,dv1,s)
        h2,dv2,s,s_ = self.d2LSTM(h1,h2,dv2,s,s_)
        y = self.classifier(h2)
        (h0,c) = hidden
        h0=h0.view(-1,self.h_dim)
        c = c.view(-1,self.h_dim)
        #print('y',y.shape,'h', h.shape)
        return y,(h0,c), h1, dv1, s, h2, dv2, s_
    
class dAlexNet(nn.Module):
    def __init__(self, h_dim,num_of_classes):
        super(dAlexNet, self).__init__()
        
        self.model = models.alexnet(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.classifier = nn.Linear(h_dim,num_of_classes)
        self.LSTM = nn.LSTM(4096,h_dim)
        self.dLSTM = dLSTM(h_dim)
        self.d2LSTM = d2LSTM(h_dim)
        self.h_dim=h_dim
        
    def forward(self,x,(h0,c),h1,dv1,s,h2,dv2,s_):
        f = self.model(x)
        f = f.view(1,-1,4096)
        hidden = (h0.view(1,-1,self.h_dim),c.view(1,-1,self.h_dim))
        out,hidden = self.LSTM(f,hidden)
        h1,dv1,s = self.dLSTM(out,h1,dv1,s)
        h2,dv2,s,s_ = self.d2LSTM(h1,h2,dv2,s,s_)
        y = self.classifier(h2)
        (h0,c) = hidden
        h0=h0.view(-1,self.h_dim)
        c = c.view(-1,self.h_dim)
        #print('y',y.shape,'h', h.shape)
        return y,(h0,c), h1, dv1, s, h2, dv2, s_

class VGG(nn.Module):
    def __init__(self, h_dim,num_of_classes):
        super(VGG, self).__init__()
        
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
        self.classifier = nn.Linear(h_dim,num_of_classes)
        self.LSTM = nn.LSTM(4096,h_dim)
        self.num_of_classes = num_of_classes
        
    def forward(self,x,hidden):
        f = self.model(x)
        f = f.view(1,-1,4096)
        out,hidden = self.LSTM(f,hidden)
        y = self.classifier(out)
        y=y.view(-1,self.num_of_classes)
        #print('y',y.shape,'h', h.shape)
        return y, hidden
class AlexNet(nn.Module):
    def __init__(self, h_dim,num_of_classes):
        super(AlexNet, self).__init__()
        
        self.model = models.alexnet(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.classifier = nn.Linear(h_dim,num_of_classes)
        self.LSTM = nn.LSTM(4096,h_dim)
        self.num_of_classes = num_of_classes
        
    def forward(self,x,hidden):
        f = self.model(x)
        f = f.view(1,-1,4096)
        out,hidden = self.LSTM(f,hidden)
        y = self.classifier(out)
        y=y.view(-1,self.num_of_classes)
        #print('y',y.shape,'h', h.shape)
        return y, hidden
        