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

        in_dim = 4096
        cat_dim = in_dim+(2*h_dim)
        self.i = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.f = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.o = nn.Sequential(nn.Linear(cat_dim,h_dim), nn.Sigmoid())
        self.g = nn.Sequential(nn.Linear(in_dim+h_dim,h_dim), nn.Tanh())

    def forward(self,x,h,dv,s):
        i = self.i(torch.cat([x,h,dv]))
        f = self.f(torch.cat([x,h,dv]))
        g = self.g(torch.cat([x,h]))
        o = self.o(torch.cat([x,h,dv]))

        s_ = torch.add(torch.mul(f,s),torch.mul(i,g))
        h = torch.mul(o,F.tanh(s_))
        dv = s_-s
        return h, dv, s_

class dVGG(nn.Module):
    def __init__(self, h_dim,num_of_classes):
        super(dVGG, self).__init__()

        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
        self.classifier = nn.Linear(h_dim,num_of_classes)
        dLSTM = dLSTM(h_dim)

    def forward(self,x,h,dv,s):
        f = self.model(x)
        h,dv,s = dLSTM(f,h,dv,s)
        y = self.classifier(h)
        return y, h, dv, s

