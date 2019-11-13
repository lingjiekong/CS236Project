import os
import sys
import torch
import numpy as np
from datasets import get_datasets, synsetid_to_cate,init_np_seed
from args import get_args
from metrics.evaluation_metrics import CD_loss
from utils import visualize_point_clouds_4
from matplotlib.pyplot import imsave

def get_train_loader(args):
    tr_dataset , _ = get_datasets(args)
    loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader

def viz_reconstruct(model,args):
    loader = iter(get_train_loader(args))
    data = next(loader)
    tr_batch =  data['train_points']
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






    



    
    