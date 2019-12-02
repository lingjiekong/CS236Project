import torch
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
#computes CH distance bwtween two point sets
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]

def CD_loss(sample_pcs,ref_pcs):
    dl, dr = distChamfer(sample_pcs, ref_pcs)
    cd_loss = dl.sum(dim=1) + dr.sum(dim=1)
    return cd_loss

def cal_mean_CD_distance(sample_pcs,ref_pcs):
    dl, dr = distChamfer(sample_pcs, ref_pcs)
    cd_loss = dl.mean(dim=1) + dr.mean(dim=1)
    return cd_loss

def cal_CD_distance(sample_pcs,ref_pcs,batch_size,avg_type='sum'):
    N_points = sample_pcs.shape[0]
    rec_loss = []
    for ii in range(0,N_points,batch_size):
        i_start = ii
        i_end = min(N_points,i_start+batch_size)
        if avg_type == 'sum':
            cd_dist = CD_loss(sample_pcs[i_start:i_end],ref_pcs[i_start:i_end])
        else:
            cd_dist = cal_mean_CD_distance(sample_pcs[i_start:i_end],ref_pcs[i_start:i_end])
        rec_loss.append(cd_dist)
    rec_loss=torch.cat(rec_loss,dim=0)
    results = {'cd_dist: ':rec_loss,'mean_dist':rec_loss.mean().item(),'std_dist':rec_loss.std().item()}
    return results

def cal_pairwise_distance(sample_pcs,ref_pcs,batch_size,avg_type='sum'):
    N_points = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    
    all_cd_list = []
    for ii in range(0,N_points):
        i_sample = sample_pcs[ii]
        cd_list = []
        for jj in range(0,N_ref,batch_size):
            i_start = jj
            i_end = min(N_ref,jj+batch_size)
            ref_batch = ref_pcs[i_start:i_end]
            cur_batch_size = ref_batch.size(0)
            sample_batch = i_sample.view(1, -1, 3).expand(cur_batch_size, -1, -1).contiguous()
            if avg_type == 'sum':
                cd_dist = CD_loss(sample_batch,ref_batch)
            else:
                cd_dist = cal_mean_CD_distance(sample_batch,ref_batch)
            cd_list.append(cd_dist)
        cd_list = torch.cat(cd_list,dim=0).unsqueeze(1)
        all_cd_list.append(cd_list)
    all_cd_list = torch.cat(all_cd_list,dim=1).t()      # 0th dimension is no. of samples and 1st dimesion is ref. samples
    #print("all_cd_list",all_cd_list.shape,"N_samples",N_points," and N_ref",N_ref)
    return all_cd_list

 #compute coverage and mmd distance
def cal_coverage_mmd(sample_pcs,ref_pcs,batch_size,avg_type='sum'):
    all_dist = cal_pairwise_distance(sample_pcs,ref_pcs,batch_size,avg_type)
    N_sample = all_dist.size(0)
    N_ref = all_dist.size(1)
    #print("Evaluation data set: ",all_dist.shape,N_sample,N_ref)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)   # calculates the min distance of each samples with the all ref samples
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    cov_mean_dist = min_val_fromsmp.mean().item()
    cov_std_dist = min_val_fromsmp.std().item()         # calculates the min distance of each ref samples with the all gen. samples
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    #print ("coverage and minimum matching distance:",cov,mmd,cov_mean_dist,cov_std_dist)
    results = {"cov":cov.item(),"cov_dist_mean":cov_mean_dist,"cov_dist_std":cov_std_dist,"mmd":mmd}
    return results

#calculate nearest neighbor
def cal_knn(sample_pcs,ref_pcs,batch_size,avg_type='sum'):
    xx = cal_pairwise_distance(ref_pcs,ref_pcs,batch_size,avg_type)
    xy = cal_pairwise_distance(sample_pcs,ref_pcs,batch_size,avg_type)
    yy = cal_pairwise_distance(sample_pcs,sample_pcs,batch_size,avg_type)
    one_nn_cd_res = knn(xx,xy,yy, 1,sqrt=False)
    #print("knn: ",one_nn_cd_res['acc'].item())
    results={'knn':one_nn_cd_res['acc'].item()}
    return results

# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx,Mxy,Myy,k,sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)
    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()
    
    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s
        

#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    in_unit_sphere=True
    sample_pcs = sample_pcs.cpu().detach().numpy()
    ref_pcs = ref_pcs.cpu().detach().numpy()
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    jsd = jensen_shannon_divergence(sample_grid_var, ref_grid_var)
    return jsd

def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    #print('====Point-clouds unit cube.',np.max(pclouds),abs(np.min(pclouds)))
    #print('=====Point-clouds  unit sphere.',np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))))
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        print('=========Point-clouds are not in unit cube.',np.max(pclouds),abs(np.min(pclouds)))

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        print('=========Point-clouds are not in unit sphere.',np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))))

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters

def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)
    return res
