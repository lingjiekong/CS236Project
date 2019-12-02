import os
import sys
import torch
import numpy as np
from datasets import get_datasets, synsetid_to_cate,init_np_seed
from args import get_args
from metrics.evaluation_metrics import CD_loss
from utils import visualize_point_clouds_4
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import random
from metrics.evaluation_metrics import cal_CD_distance,cal_coverage_mmd,cal_knn,jsd_between_point_cloud_sets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_loader(args):
    tr_dataset , _ = get_datasets(args)
    loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader

def get_test_loader(args,init_seed=2019):
    _,te_dataset = get_datasets(args)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,worker_init_fn=init_seed)
    return loader

def viz_reconstruct(model,args,nset=2,dtype='test',viz_size=8 ):
    save_dir = "viz_reconstruct"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    temp_var=args.batch_size
    args.batch_size=viz_size
    if dtype=='train':
        loader = get_train_loader(args)
    else:
        loader = get_test_loader(args)
    with torch.no_grad():
        loss=[]
        data_num=0
        for counter,data in enumerate(loader):
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            te_pc =  te_pc.to(device)
            samples = model.reconstruct_input(te_pc)
            g_truth=CD_loss(te_pc,samples)
            loss.append(g_truth)
            # save viz
            if te_pc.shape[0] ==viz_size:
                results = []
                for idx in range(0,viz_size,2):
                    res = visualize_point_clouds_4(samples[idx],te_pc[idx],samples[idx+1],te_pc[idx+1],
                    data_num,data_num+1,g_truth[idx].item(),g_truth[idx+1].item(),[0, 2, 1])
                    results.append(res)
                    res = np.concatenate(results, axis=1)
                    data_num+=2
                imsave(save_dir+'/reconstruct_'+str(counter)+'.png', res.transpose((1, 2, 0)))
        loss=torch.cat(loss,axis=0).numpy()
        np.save("reconstruction_loss",loss)
        print("Number of examples: ",loss.shape[0])
        print("Mean and Std CD Loss: ",loss.mean(),loss.std())
        print("Min and max loss:",np.min(loss),np.max(loss),"at",np.argmin(loss),np.argmax(loss))
        plt.figure()
        plt.title("CD loss Histogram")
        plt.hist(loss,bins=20)
        plt.savefig("cd_loss")
    args.batch_size=temp_var
    return

def sample_structure(model,nset=2,b_size=16):
    save_dir = "viz_sample" 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for jdx in range(nset):
            samples = model.sample_point(b_size)
            results = []
            for idx in range(0,b_size,4):
                res = visualize_point_clouds_4(samples[idx],samples[idx+1],
                samples[idx+2],samples[idx+3],idx,idx+1,0,0,[0, 2, 1],tag=False)
                results.append(res)
                res = np.concatenate(results, axis=1)
            imsave(save_dir+'/samples_'+str(jdx)+'.png', res.transpose((1, 2, 0)))

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

#evaluate the reconstructed data distribution with the true data dristribution
def eval_model_reconstruct(model,args,denormalize=True):
    model.eval()
    loader = get_test_loader(args)
    true_sample = []
    gen_sample = []
    print("args.resume_dataset_mean:",args.resume_dataset_mean)
    with torch.no_grad():
        for data in loader:
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            samples = model.reconstruct_input(te_pc)
            if denormalize==True:                         #denormalize = True is required for computing JS-Divergence (makes data between 0-1)
                m, s = data['mean'].float(), data['std'].float()
                m = m.to(device)
                s = s.to(device)
                te_pc = te_pc * s + m
                samples = samples * s + m
            true_sample.append(te_pc)
            gen_sample.append(samples)
        true_sample = torch.cat(true_sample,axis=0)
        gen_sample = torch.cat(gen_sample,axis=0)
        compute_metrics(gen_sample,true_sample,args,denormalize,cal_cd_dist=True)

#evaluate the generated data distribution with the true data dristribution
def eval_model_random_sample(model,args,dtype='test',Nsamples=None,denormalize=True):
    model.eval()
    if dtype=='train':
        loader = get_train_loader(args)
    else:
        loader = get_test_loader(args)
    true_sample = []
    gen_sample = []
    m_all = []
    s_all = []
    with torch.no_grad():
        for data in loader:
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            if denormalize == True:                 #denormalize = True is required for computing JS-Divergence (makes data between 0-1)
                m, s = data['mean'].float(), data['std'].float()
                m = m.to(device)
                s = s.to(device)
                m_all.append(m)
                s_all.append(s)
                te_pc = te_pc * s + m
            true_sample.append(te_pc)
        true_sample = torch.cat(true_sample,axis=0)
        if denormalize == True:
            m_all = torch.cat(m_all,axis=0)
            s_all = torch.cat(s_all,axis=0)
        if Nsamples==None:
            Nsamples = true_sample.size(0)
        for ii in range(0,Nsamples,args.batch_size):
            i_points=min(args.batch_size,Nsamples-ii)
            samples=model.sample_point(i_points)
            if denormalize == True:
                if ii <= true_sample.size(0):
                    m =  m_all[ii:min(ii+args.batch_size,true_sample.size(0))]
                    s =  s_all[ii:min(ii+args.batch_size,true_sample.size(0))]
                else:
                    m = m_all[0:min(args.batch_size,Nsamples-ii)]
                    s = s_all[0:min(args.batch_size,Nsamples-ii)]
                samples = samples * s + m
            gen_sample.append(samples)
        gen_sample = torch.cat(gen_sample,axis=0)
        compute_metrics(gen_sample,true_sample,args,denormalize,cal_cd_dist=False)

#calculates Covarage, Minimum Matching Distance , 1-st nearest neighbor and JS-Divergance between generated and true data distribution
def compute_metrics(gen_sample,true_sample,args,denormalize,cal_cd_dist=False):
    N_sample = gen_sample.shape[0]
    N_ref = true_sample.shape[0]
    print("No. of gen samples: ",gen_sample.shape)
    print("No. of. true sample", true_sample.shape)
    if (N_sample==N_ref and cal_cd_dist==True):
        #cal_cd_dist for generated random data is not useful metric. only use covarage, miminum matching distrace and JS Divergence
        print ("=========Reconstruction loss using CD Distsance ==========")
        ret = cal_CD_distance(gen_sample,true_sample,args.batch_size)
        print("mean and std distance: ",ret['mean_dist'],ret['std_dist'])
    print("=======Calculate Covrage and Minimum Matching Distance=======")
    ret = cal_coverage_mmd(gen_sample,true_sample,args.batch_size)
    print("Coverage {} , mean distance {}, std {}".format(ret['cov'],ret['cov_dist_mean'],ret['cov_dist_std']))
    print('Minimum Matching Distance {}'.format(ret['mmd']))
    if (N_sample==N_ref):
        print("=====calculate 1st-nearest neighbor metric=====")  # we want value closer to 0.5
        ret = cal_knn(gen_sample,true_sample,args.batch_size)
        print("1st-nearest neighbor's accuracy {}".format(ret['knn']))
    if denormalize == True:
        print("====Calculate JSD between point clouds=====")
        jsd = jsd_between_point_cloud_sets(gen_sample,true_sample)
        print("jsd: ",jsd)
    return

#compute nelbo of the model on test data
def cal_nelbo_samples(model,args):
    temp_var=args.batch_size
    args.batch_size=1
    loader = get_test_loader(args)
    loss=[]
    model.eval()
    kl_loss=[]
    nelbo=[]
    rec_loss=[]
    with torch.no_grad():
        for data in loader:
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            inputs = te_pc.to(device)
            inputs_dict = {'x':inputs}
            if model.type == 'CVAE':
                n_class = len(args.cates)
                obj_type = data['cate_idx']
                y_one_hot = obj_type.new(np.eye(n_class)[obj_type]).float()
                inputs_dict['y_class'] = y_one_hot.to(device)
            ret = model(inputs_dict)
            nelbo.append(ret['nelbo'].item())
            kl_loss.append(ret['kl_loss'].item())
            rec_loss.append(ret['x_reconst'].item())
    print(len(nelbo))
    nelbo=np.asarray(nelbo).reshape(-1)
    kl_loss=np.asarray(kl_loss).reshape(-1)
    rec_loss=np.asarray(rec_loss).reshape(-1)
    print("======Likelihood of the model on test data========")
    print("Numboer of test examples: ",nelbo.shape)
    avg_nelbo, std_nelbo =nelbo.mean(),nelbo.std()
    avg_kl_loss, std__kl_loss = kl_loss.mean(),kl_loss.std()
    avg_rec_loss, std__rec_loss =rec_loss.mean(), rec_loss.std()
    print("Nelbo: mean {} std {} minimum {} maximum {}".format(avg_nelbo, std_nelbo,np.min(nelbo),np.max(nelbo)))
    print("kl_loss  mean {} std {} minimum {} maximum {} ".format(avg_kl_loss, std__kl_loss,np.min(kl_loss),np.max(kl_loss)))
    print("Rec_loss  mean {} std {} minimum {} maximum {}".format(avg_kl_loss, std__kl_loss,np.min(rec_loss),np.max(rec_loss)))
    plt.figure()
    plt.title("NELBO Histogram")
    plt.hist(nelbo,bins=20)
    plt.savefig("nelbo")
    plt.figure()
    plt.title("KL Divergence Histogram")
    plt.hist(kl_loss,bins=20)
    plt.savefig("kl_loss")
    plt.figure()
    plt.title("Reconstruction loss Histogram")
    plt.hist(rec_loss,bins=20)
    plt.savefig("Rec_loss")
