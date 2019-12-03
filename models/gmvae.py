import torch
from torch import nn
import torch.nn.functional as F
import utils as ut
import numpy as np
from metrics.evaluation_metrics import CD_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pudb

class GMVAE(nn.Module):
    def __init__(self,k,encoder,decoder,z_classifer,args):
        super(GMVAE, self).__init__()
        self.k = k
        self.n_point = args.tr_max_sample_points
        self.point_dim = 3
        self.n_point_3 = self.point_dim * self.n_point 
        self.z_dim = args.zdim
        self.loss_type = 'chamfer'
        self.loss_sum_mean = args.loss_sum_mean
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.use_encoding_in_decoder = args.use_encoding_in_decoder
        self.encoder = encoder(self.z_dim,self.point_dim,self.use_deterministic_encoder)
        
        if not self.use_deterministic_encoder and self.use_encoding_in_decoder:
            self.decoder = decoder(2 *self.z_dim,self.n_point,self.point_dim)
        else:
            self.decoder = decoder(self.z_dim,self.n_point,self.point_dim)
        self.z_classifer = z_classifer(2*self.z_dim, len(args.cates))
        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)
        self.type = 'GMVAE'
    
    def forward(self, inputs):
        ret = {}
        x, y_class = inputs['x'], inputs['y_class']
        m, v = self.encoder(x)
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        if self.use_deterministic_encoder:
            y = self.decoder(m)
            kl_loss = torch.zeros(1)
        else:
            z =  ut.sample_gaussian(m,v)
            decoder_input = z if not self.use_encoding_in_decoder else \
            torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
            y = self.decoder(decoder_input)
            #compute KL divergence loss :
            z_prior_m, z_prior_v = prior[0], prior[1]
            kl_loss = ut.log_normal(z, m, v) - ut.log_normal_mixture(z, z_prior_m, z_prior_v)
        #compute reconstruction loss 
        if self.loss_type is 'chamfer':
            x_reconst = CD_loss(y,x)
        # mean or sum
        if self.loss_sum_mean == "mean":
            x_reconst = x_reconst.mean()
            kl_loss = kl_loss.mean()
        else:
            x_reconst = x_reconst.sum()
            kl_loss = kl_loss.sum()
        nelbo = x_reconst + kl_loss
        ret = {'nelbo':nelbo, 'kl_loss':kl_loss, 'x_reconst':x_reconst}
        # classifer network
        mv = torch.cat((m,v),dim=1)
        y_logits = self.z_classifer(mv)
        z_cl_loss = self.z_classifer.cross_entropy_loss(y_logits,y_class)
        ret['z_cl_loss'] = z_cl_loss
        return ret
    
    def sample_point(self,batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        z = ut.sample_gaussian(m, v)
        decoder_input = z if not self.use_encoding_in_decoder else \
        torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
        y = self.sample_x_given(decoder_input)
        return y

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

    def compute_sigmoid_given(self, z):
        logits = self.decoder(z)
        return torch.sigmoid(logits)

    def reconstruct_input(self,x):
        m, v = self.encoder(x)
        if self.use_deterministic_encoder:
            y = self.decoder(m)
        else:
            z =  ut.sample_gaussian(m,v)
            decoder_input = z if not self.use_encoding_in_decoder else \
            torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
            y = self.sample_x_given(decoder_input)
        return y
