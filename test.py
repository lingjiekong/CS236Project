import os
import sys
import torch
import numpy as np
from datasets import get_datasets, synsetid_to_cate,init_np_seed
from args import get_args
from metrics.evaluation_metrics import CD_loss
from utils import visualize_point_clouds_4
from matplotlib.pyplot import imsave
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_loader(args):
    tr_dataset , _ = get_datasets(args)
    loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader

def viz_reconstruct(model,args):
    loader = iter(get_train_loader(args))
    for i in range(np.random.randint(1,10)):
        data = next(loader)
    tr_batch =  data['train_points'].to(device)
    with torch.no_grad():
        samples = model.reconstruct_input(tr_batch)
        g_truth=CD_loss(tr_batch,samples)
        print("Difference: ",g_truth)
        results = []
        for idx in range(0,8,2):
            res = visualize_point_clouds_4(samples[idx],tr_batch[idx],samples[idx+1],tr_batch[idx+1],
                   idx,idx+1,[0, 2, 1])
            results.append(res)
        res = np.concatenate(results, axis=1)
        imsave('reconstruct.png', res.transpose((1, 2, 0)))

def sample_structure(model):
    with torch.no_grad():
        samples = model.sample_point(64)
        results = []
        for idx in range(0,16,4):
            res = visualize_point_clouds_4(samples[idx],samples[idx+1],
            samples[idx+2],samples[idx+3],idx,idx+1,[0, 2, 1],tag=False)
            results.append(res)
        res = np.concatenate(results, axis=1)
        imsave('samples.png', res.transpose((1, 2, 0)))

def evaluate_model(model, dataset, args, init_seed=2019, batch_size=64):
    total_reconstruct_loss = 0
    model.eval()    
    data_iter = torch.utils.data.DataLoader(
                                dataset=dataset, 
                                batch_size= batch_size, 
                                shuffle=False,
                                num_workers=0, 
                                pin_memory=True, 
                                drop_last=False,
                                worker_init_fn=init_seed)
    with torch.no_grad():
        for bidx, data in enumerate(data_iter):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']

#             if args.random_rotate:
#                 tr_batch, _, _ = apply_random_rotation(
#                     tr_batch, rot_axis=train_loader.dataset.gravity_axis)

            inputs = tr_batch.to(device)
            inputs_dict = {'x':inputs}

            if model.type == 'CVAE':
                n_class = len(args.cates)
                obj_type = data['cate_idx']
                y_one_hot = obj_type.new(np.eye(n_class)[obj_type]).float()   
                inputs_dict['y_class'] = y_one_hot.to(device)
                
            ret = model(inputs_dict)
            x_reconst = ret['x_reconst']

            cur_x_reconst = x_reconst.cpu().item()
            total_reconstruct_loss += cur_x_reconst
        return total_reconstruct_loss / (bidx+1)




    



    
    