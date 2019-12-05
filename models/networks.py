import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

#simple encodr design
class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)
            v = F.softplus(v) + 1e-8

        return m, v

#decoder design
class MLP_Decoder(nn.Module):
    def __init__(self,zdim,n_point,point_dim):
        super(MLP_Decoder,self).__init__()
        self.zdim = zdim
        self.n_point = n_point
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.fc1 = nn.Linear(self.zdim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.final = nn.Linear(256,self.n_point_3)
    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        output  =  self.final(x)
        output = output.reshape(-1,self.n_point,self.point_dim)
        return output

class MLP_Conv_v1(nn.Module):
    def __init__(self,zdim,n_point,point_dim):
        super(MLP_Conv_v1,self).__init__()
        self.zdim = zdim
        self.n_point = n_point
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.fc1 = nn.Linear(self.zdim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_point_3)
        self.conv1 = nn.Conv1d(self.point_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, self.point_dim, 1)
    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.reshape(-1,self.point_dim,self.n_point)
        x = F.relu(self.bn1(self.conv1(x)))
        output = self.conv2(x).transpose(1,2)
        return output

class MLP_Conv_v2(nn.Module):
    def __init__(self,zdim,n_point,point_dim):
        super(MLP_Conv_v2,self).__init__()
        self.zdim = zdim
        self.n_point = n_point
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.fc1 = nn.Linear(self.zdim, self.n_point_3)
        self.conv1 = nn.Conv1d(self.point_dim, 256, 1)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.conv3 = nn.Conv1d(128, self.point_dim, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = x.reshape(-1,self.point_dim,self.n_point)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        output = self.conv3(x).transpose(1,2)
        return output



#encoder design for decoder 
#input pointset is 1024
class Decoder_condtional_x(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(Decoder_condtional_x, self).__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        #get the compressed z-vector for the input point cloud
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, zdim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1(self.fc1(x)))
        m = F.relu(self.fc_bn2(self.fc2(m)))
        m = self.fc3(m)
        return m

#only partition input 2048 points into 2 sets of 1024 points
class autoregressive_decoder(nn.Module):
    def __init__(self,zdim,n_point,point_dim,add_latent_var=True):
        super(autoregressive_decoder,self).__init__()
        self.zdim = zdim
        self.n_point = int(n_point/2)
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.decoder_0 = MLP_Decoder(self.zdim,self.n_point,self.point_dim)
        self.encoder = Decoder_condtional_x(self.zdim,self.point_dim)
        self.add_latent_var = add_latent_var
        if self.add_latent_var == True:
            self.decoder_1 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        else:
            self.decoder_1 = MLP_Decoder(self.zdim,self.n_point,self.point_dim)
    def forward(self,z):
        x_0 = self.decoder_0(z)
        z_0 = self.encoder(x_0)
        if self.add_latent_var == True:
            z_1 = torch.cat([z,z_0],dim=1)
        else:
            z_1 = z_0
        x_1 = self.decoder_1(z_1)
        x = torch.cat([x_0,x_1],dim=1)
        return x

#partition input 2048 points into m groups each with set of n_i points
class autoregressive_decoder_multi(nn.Module):
    def __init__(self,zdim,n_point,point_dim,point_group=8,add_latent_var=True):
        super(autoregressive_decoder_multi,self).__init__()
        self.zdim = zdim
        self.point_group = point_group
        self.n_point = int(n_point/self.point_group)
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.decoder_0 = MLP_Decoder(self.zdim,self.n_point,self.point_dim)
        self.encoder = Decoder_condtional_x(self.zdim,self.point_dim)
        self.add_latent_var = add_latent_var
        if self.add_latent_var == True:
            self.decoder_1 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        else:
            self.decoder_1 = MLP_Decoder(self.zdim,self.n_point,self.point_dim)
    def forward(self,z):
        x = self.decoder_0(z)
        z_0 = self.encoder(x)
        for i in range(self.point_group-1):
            if self.add_latent_var == True:
                z_1 = torch.cat([z,z_0],dim=1)
            else:
                z_1 = z_0
            x_1 = self.decoder_1(z_1)
            z_2 = self.encoder(x_1)
            z_0 = z_1[:,self.zdim:] + z_2 
            x = torch.cat([x,x_1],dim=1)
            #print("x:",x.shape,self.point_group,self.n_point)
        #print("x final:",x.shape,self.point_group,self.n_point)
        #exit(1)
        return x

#partition input 2048 points into m groups each with set of n_i points
class autoregressive_decoder_multi_decoder(nn.Module):
    def __init__(self,zdim,n_point,point_dim,point_group=8,add_latent_var=True):
        super(autoregressive_decoder_multi_decoder,self).__init__()
        self.zdim = zdim
        self.point_group = point_group
        self.n_point = int(n_point/self.point_group)
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.decoder_0 = MLP_Decoder(self.zdim,self.n_point,self.point_dim)
        self.encoder = Decoder_condtional_x(self.zdim,self.point_dim)
        self.add_latent_var = add_latent_var

        self.decoder_1_1 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        self.decoder_1_2 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        self.decoder_1_3 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        self.decoder_1_4 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        self.decoder_1_5 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        self.decoder_1_6 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        self.decoder_1_7 = MLP_Decoder(2*self.zdim,self.n_point,self.point_dim)
        
    def forward(self,z):
        x = self.decoder_0(z)
        z_0 = self.encoder(x)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_1(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_2(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_3(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_4(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_5(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_6(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)

        z_1 = torch.cat([z,z_0],dim=1)
        x_1 = self.decoder_1_7(z_1)
        z_2 = self.encoder(x_1)
        z_0 = z_1[:,self.zdim:] + z_2 
        x = torch.cat([x,x_1],dim=1)


            #print("x:",x.shape,self.point_group,self.n_point)
        #print("x final:",x.shape,self.point_group,self.n_point)
        #exit(1)
        return x

#    def forward(self,z):
#        x = self.decoder_0(z)
#        for i in range(self.point_group-1):
#            z_0 = self.encoder(x)
#            if self.add_latent_var == True:
#                z_1 = torch.cat([z,z_0],dim=1)
#            else:
#                z_1 = z_0
#            x_1 = self.decoder_1(z_1)
#            x = torch.cat([x,x_1],dim=1)
#        return x










