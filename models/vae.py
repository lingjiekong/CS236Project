import torch
from torch import nn
import torch.nn.functional as F
import utils as ut
import numpy as np
from metrics.evaluation_metrics import CD_loss

class VAE(nn.Module):
    def __init__(self,encoder,decoder,args):
        super(VAE, self).__init__()
        self.n_point = args.tr_max_sample_points
        self.point_dim = 3
        self.n_point_3 = self.point_dim * self.n_point 
        self.z_dim = args.zdim
        self.loss_type = 'chamfer'
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.encoder = encoder(self.z_dim,self.point_dim,self.use_deterministic_encoder)
        self.decoder = decoder(self.z_dim,self.n_point,self.point_dim)
        #set prior parameters of the vae model p(z)
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
    
    def forward(self,x):
        m, v = self.encoder(x)
        if self.use_deterministic_encoder:
            y = self.decoder(m)
            kl_loss = torch.zeros(1)
        else:
            z =  ut.sample_gaussian(m,v)
            y = self.decoder(z)
            #compute KL divergence loss :
            p_m = self.z_prior[0].expand(m.size())
            p_v = self.z_prior[1].expand(v.size())
            kl_loss = ut.kl_normal(m,v,p_m,p_v)
        #compute reconstruction loss 
        if self.loss_type is 'chamfer':
            x_reconst = CD_loss(y,x)
        x_reconst = x_reconst.mean()
        kl_loss = kl_loss.mean()
        nelbo = x_reconst + kl_loss
        return nelbo,kl_loss,x_reconst
    

    def sample_point(self,batch):
        p_m = self.z_prior[0].expand(batch,self.z_dim)
        p_v = self.z_prior[1].expand(batch,self.z_dim)
        z =  ut.sample_gaussian(p_m,p_v)
        y = self.decoder(z)
        return y

    def reconstruct_input(self,x):
        m, v = self.encoder(x)
        if self.use_deterministic_encoder:
            y = self.decoder(m)
        else:
            z =  ut.sample_gaussian(m,v)
            y = self.decoder(z)
        return y
