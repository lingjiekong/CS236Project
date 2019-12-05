import torch
from torch import nn
import torch.nn.functional as F
import utils as ut
import numpy as np
from models.flow import InverseAutoregressiveFlow, Reverse, FlowSequential 
from metrics.evaluation_metrics import CD_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import pudb

class Flow_Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(Flow_Encoder, self).__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

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

        #Get context vector 
        self.fc1_h = nn.Linear(512, 256)
        self.fc2_h = nn.Linear(256, 128)
        self.fc3_h = nn.Linear(128, zdim)
        self.fc_bn1_h = nn.BatchNorm1d(256)
        self.fc_bn2_h = nn.BatchNorm1d(128)


    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)
        v = F.softplus(v) + 1e-8
        h = F.relu(self.fc_bn1_h(self.fc1_h(x)))
        h = F.relu(self.fc_bn2_h(self.fc2_h(h)))
        h = self.fc3_h(h)

        return m, v, h


class VAE_Flow(nn.Module):
    def __init__(self,encoder,decoder,args,flow_depth=5):
        super(VAE_Flow, self).__init__()
        self.n_point = args.tr_max_sample_points
        self.point_dim = 3
        self.n_point_3 = self.point_dim * self.n_point 
        self.z_dim = args.zdim
        self
        self.loss_type = 'chamfer'
        self.encoder = encoder(self.z_dim,self.point_dim)
        self.flow_depth = flow_depth
        self.flow_layer  = []
        self.decoder = decoder(self.z_dim,self.n_point,self.point_dim)

        for _ in range(self.flow_depth):
            self.flow_layer.append(InverseAutoregressiveFlow(num_input=self.z_dim,num_hidden=self.z_dim*2,num_context=None))
            self.flow_layer.append(Reverse(self.z_dim))
        self.q_z_flow = FlowSequential(*self.flow_layer)

        #set prior parameters of the vae model p(z)
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.type = 'Flow_VAE'
    
    def forward(self, inputs):
        x = inputs['x']
        pi = torch.tensor(2.0 * 3.14159265359)
        m, v, h = self.encoder(x)
        epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size())).to(device)
        z_0 = m + torch.sqrt(v) * epsilon
        log_q_z_0 = 0.5*epsilon*epsilon + torch.log(torch.sqrt(v)) + 0.5*torch.log(pi)
        log_q_z_0 = -log_q_z_0.sum(-1)
        #z_0 =  ut.sample_gaussian(m,v)
        #log_q_z_0 = ut.log_normal(z_0,m,v)
        z_T, log_q_z_flow = self.q_z_flow(z_0,context=None)
        log_q_z = log_q_z_0 + log_q_z_flow.sum(-1)
        y = self.decoder(z_T)

        #compute log prior(z_T) :
        p_m = self.z_prior[0].expand(m.size())
        p_v = self.z_prior[1].expand(v.size())
        log_prior_z_T = ut.log_normal(z_T,p_m,p_v)
        #compute reconstruction loss 
        if self.loss_type is 'chamfer':
            x_reconst = CD_loss(y,x)
        
        x_reconst = x_reconst.mean()
        log_prior_z_T = log_prior_z_T.mean()
        log_q_z = log_q_z.mean()
        nelbo = x_reconst - log_prior_z_T + log_q_z

        ret = {'nelbo':nelbo, 'kl_loss':torch.zeros(1), 'x_reconst':x_reconst, 'flow_loss': log_q_z,'log_prior_z_T':log_prior_z_T}
        #print("loss: ",nelbo.item(),x_reconst.item(),-log_prior_z_T.item(),log_q_z.item())
        #exit(1)
        return ret
    

    def sample_point(self,batch):
        p_m = self.z_prior[0].expand(batch,self.z_dim).to(device)
        p_v = self.z_prior[1].expand(batch,self.z_dim).to(device)
        z_0 =  ut.sample_gaussian(p_m,p_v)
        z_T, _ = self.q_z_flow(z_0,context=None)
        y = self.decoder(z_T)
        return y

    def reconstruct_input(self,x):
        m, v, h = self.encoder(x)
        epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size())).to(device)
        z_0 = m + torch.sqrt(v) * epsilon
        z_T, _ = self.q_z_flow(z_0,context=None)
        y = self.decoder(z_T)
        return y
