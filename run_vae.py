import torch
import numpy as np
from args import get_args
from models.vae import VAE
from models.gmvae import GMVAE
from models.networks import Encoder, MLP_Decoder, MLP_Conv_v1, MLP_Conv_v2, Classifier, ZClassifier
from utils import set_random_seed
from train import train
from test  import viz_reconstruct, sample_structure
from datasets import get_datasets, init_np_seed
from test import evaluate_model, evaluate_classifer_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = get_args()

args.zdim = 128
args.batch_size = 16
args.lr = 2e-3
args.tr_max_sample_points = 2048
args.data_dir="data/ShapeNetCore.v2.PC15k"
args.loss_sum_mean = "mean" # can be also "mean"

set_random_seed(args.seed)

print("args.epochs",args.epochs,args.log_freq,args.random_rotate)

encoder = Encoder
decoder = MLP_Conv_v1  #MLP_Decoder
z_classifer = ZClassifier
# model = VAE(encoder,decoder,z_classifer,args)
k = 10
model = GMVAE(10, encoder, decoder, z_classifer, args)
if device.type == 'cuda':
    model = model.cuda()
    
#train model
if args.train_model == 1:
    # args.log_name = "vae"
    # args.train_model_name = "vae"
    args.log_name = "gmvae"
    args.train_model_name = "gmvae"
    train(model,args)    

#evaliuate a trained model
if args.train_model == 0:
    args.log_name = "gmvae"
    args.train_model_name = "gmvae"
    # args.log_name = "vae"
    # args.train_model_name = "vae"
    args.resume_checkpoint='checkpoints/'+args.log_name+'/checkpoint-latest.pt'
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    viz_reconstruct(model,args)
    if args.use_deterministic_encoder==False:
        sample_structure(model)

# train z classifier
if args.train_model == 3:
    args.log_name = "classifier"
    args.train_model_name = "gmvae"
    # args.log_name = "classifier"
    # args.train_model_name = "vae"
    train(model,args)

# eval z classifier
if args.train_model == 2:
    # args.log_name = "classifier"
    # args.train_model_name = "vae"
    args.log_name = "classifier"
    args.train_model_name = "gmvae"
    args.resume_checkpoint='checkpoints/'+args.log_name+'/checkpoint-latest.pt'
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    #initilize dataset and load
    tr_dataset, te_dataset = get_datasets(args)
    eval_metric = evaluate_classifer_model(model, te_dataset, args)
    train_metric = evaluate_classifer_model(model, tr_dataset, args)
    print('Checkpoint: Dev Loss:{0}, Train Loss:{1}'.format(eval_metric, train_metric))
    