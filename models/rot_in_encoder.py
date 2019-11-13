import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from models.networks import Encoder

#Borrowed from https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3,64,1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self,x):
        batchsize = x.size()[0]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class Rot_Encoder(nn.Module):
    def __init__(self,zdim,point_dim,use_deterministic_encoder=False):
        super(Rot_Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.point_dim = point_dim
        self.transform = STN3d()
        self.feature = Encoder(self.zdim,self.point_dim,self.use_deterministic_encoder)
    
    def forward(self,x):
        rot_matix = self.transform(x)
        x = torch.bmm(x,rot_matix)
        m, v = self.feature(x)
        return m,v












