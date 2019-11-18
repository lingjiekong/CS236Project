import sys
import os
import torch 
import torch.distributed as dist
from torch import optim
import torch.nn as nn
import numpy as np
import random
from utils import set_random_seed,visualize_point_clouds,save,resume,apply_random_rotation
from datasets import get_datasets, init_np_seed
from matplotlib.pyplot import imsave


def initilize_optimizer(model,args):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        assert 0, "args.optimizer should be either 'adam' or 'sgd'"
    return optimizer
    

def main_train_loop(save_dir,ngpus_per_node,model,args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_class = len(args.cates)
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    #resume chekckpoint
    start_epoch = 0
    optimizer=initilize_optimizer(model,args)
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict))
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))
        print('Resumed from: ' + args.resume_checkpoint)
    
    #initilize dataset and load
    tr_dataset, te_dataset = get_datasets(args)

    train_sampler = None   # for non distributed training

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)
    
    #initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"
    
    #training starts from here
    tot_nelbo=[]
    tot_kl_loss=[]
    tot_x_reconst=[]


    for epoch in range(start_epoch,args.epochs):
        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)
        #train for one epoch
        model.train()
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            obj_type = data['cate_idx']
            y_one_hot = obj_type.new(np.eye(n_class)[obj_type]).to(device).float()
            step = bidx + len(train_loader) * epoch

            if args.random_rotate:
                tr_batch, _, _ = apply_random_rotation(
                    tr_batch, rot_axis=train_loader.dataset.gravity_axis)

            if torch.cuda.is_available():
                inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            else:
                inputs = tr_batch

            optimizer.zero_grad()
            loss, nelbo, kl_loss, x_reconst, cl_loss = model(tr_batch,y_one_hot)
            loss.backward()
            optimizer.step()

            cur_loss = loss.cpu().item()
            cur_nelbo= nelbo.cpu().item()
            cur_kl_loss = kl_loss.cpu().item()
            cur_x_reconst = x_reconst.cpu().item()
            cur_cl_loss = cl_loss.cpu().item()
            tot_nelbo.append(cur_nelbo)
            tot_kl_loss.append(cur_kl_loss)
            tot_x_reconst.append(cur_x_reconst)
            if step % args.log_freq == 0:
                print("Epoch {0:6d} Step {1:12d} Loss {2:12.6f} Nelbo {3:12.6f} KL Loss {4:12.6f} Reconst Loss {5:12.6f} CL_Loss{6:12.6f}"
                .format(epoch,step,cur_loss,cur_nelbo,cur_kl_loss,cur_x_reconst,cur_cl_loss))
        
        #save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save(model, optimizer, epoch + 1,os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1,os.path.join(save_dir, 'checkpoint-latest.pt'))

    save(model, optimizer, args.epochs,os.path.join(save_dir, 'checkpoint-latest.pt'))
    #save final visuliztion of 10 samples
    model.eval()
    with torch.no_grad():
        samples_A = model.reconstruct_input(tr_batch)  #sample_point(5)
        results = []
        for idx in range(5):
            res = visualize_point_clouds(samples_A[idx],tr_batch[idx],idx,
                    pert_order=train_loader.dataset.display_axis_order)
            results.append(res)
        res = np.concatenate(results, axis=1)
        imsave(os.path.join(save_dir, 'images', '_epoch%d.png' % (epoch)), res.transpose((1, 2, 0)))


def train(model,args):
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'images'))
        
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely ' 'disable data parallelism.')
    
    print("--------Arguments--------")
    print(args)
    print("--------------------------")

    ngpus_per_node = torch.cuda.device_count()
    main_train_loop(save_dir,ngpus_per_node,model,args)
    return
