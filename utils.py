import numpy as np
from math import log, pi
import os
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def safe_log(z):
    return torch.log(z + 1e-7)

def sample_gaussian(m, v):
    epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size())).to(device)
    z = m + torch.sqrt(v) * epsilon
    return z

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    # print ("[x, m, v]: {}, {}, {}".format(x.shape, m.shape, v.shape))
    element_log_prob = -0.5*(np.log(np.pi*2) + torch.log(v) + (x-m).pow(2)/v)
    log_prob = element_log_prob.sum(-1)

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################
    z = z.unsqueeze(1)
    log_prob = log_normal(z, m, v)
    log_prob = log_mean_exp(log_prob, 1)

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))

def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()

def kl_normal(qm,qv,pm,pv):
    # tensor shape (Batch,dim)
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

# Augmentation
def apply_random_rotation(pc, rot_axis=1):
    B = pc.shape[0]

    theta = np.random.rand(B) * 2 * np.pi
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    if rot_axis == 0:
        rot = np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 1:
        rot = np.stack([
            cos, zeros, -sin,
            zeros, ones, zeros,
            sin, zeros, cos
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 2:
        rot = np.stack([
            ones, zeros, zeros,
            zeros, cos, -sin,
            zeros, sin, cos
        ]).T.reshape(B, 3, 3)
    else:
        raise Exception("Invalid rotation axis")
    rot = torch.from_numpy(rot).to(pc)

    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    pc_rotated = torch.bmm(pc, rot)
    return pc_rotated, rot, theta

def save(model, optimizer, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(d, path)

def resume(path, model, optimizer=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, start_epoch

def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True 
# Visualization
def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res

def visualize_point_clouds_4(pts_1, gtr_1, pts_2, gtr_2,idx_1, idx_2,pert_order=[0, 1, 2],tag=True):
    pts_1 = pts_1.cpu().detach().numpy()[:, pert_order]
    gtr_1 = gtr_1.cpu().detach().numpy()[:, pert_order]
    pts_2 = pts_2.cpu().detach().numpy()[:, pert_order]
    gtr_2 = gtr_2.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(pts_1[:, 0], pts_1[:, 1], pts_1[:, 2], s=5)

    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(gtr_1[:, 0], gtr_1[:, 1], gtr_1[:, 2], s=5)

    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(pts_2[:, 0], pts_2[:, 1], pts_2[:, 2], s=5)

    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(gtr_2[:, 0], gtr_2[:, 1], gtr_2[:, 2], s=5)

    if tag:
        ax1.set_title("Sample:%s" % idx_1)
        ax2.set_title("Ground Truth:%s" % idx_1)
        ax3.set_title("Sample:%s" % idx_2)
        ax4.set_title("Ground Truth:%s" % idx_2)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res

