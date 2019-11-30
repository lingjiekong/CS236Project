import torch
import numpy as np
from args import get_args
from models.cvae import CVAE
from models.networks import Encoder, MLP_Decoder, MLP_Conv_v1, MLP_Conv_v2, Classifier
from cvae_train import train
from utils import set_random_seed
from test  import viz_reconstruct, sample_structure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = get_args()
#overwrite selected default parameters
args.zdim = 128
args.batch_size = 64
args.lr = 2e-3
args.tr_max_sample_points = 2048
args.data_dir="data/ShapeNetCore.v2.PC15k"
args.log_name = "cvae"

set_random_seed(args.seed)

print("args.epochs",args.epochs,args.log_freq,args.random_rotate)

encoder = Encoder
decoder = MLP_Decoder  # MLP_Conv_v1  #MLP_Decoder
classifer = Classifier
    
model = CVAE(encoder,decoder,classifer,args)

if device.type == 'cuda':
    model.cuda()
#train model
if args.train_model == 1:
   train(model,args)

#evaliuate a trained model
if args.train_model == 0:
    args.resume_checkpoint='checkpoints/'+args.log_name+'/checkpoint-latest.pt'
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    viz_reconstruct(model,args)
    if args.use_deterministic_encoder==False:
        sample_structure(model)
