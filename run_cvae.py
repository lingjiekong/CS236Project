import torch
import numpy as np
from args import get_args
from models.cvae import CVAE
from models.networks import Encoder, MLP_Decoder, MLP_Conv_v1, MLP_Conv_v2
from cvae_train import train
from test  import viz_reconstruct, sample_structure

args = get_args()
#overwrite selected default parameters
args.cates = ['airplane','chair']     #chair
args.zdim = 128
args.batch_size = 64
args.lr = 2e-3
args.epochs = 400
args.tr_max_sample_points = 2048
args.data_dir="data/ShapeNetCore.v2.PC15k"

#args.random_rotate = True
args.log_freq=100
args.use_deterministic_encoder = False     #AE
args.log_name = 'cvae'
#args.train_model = 1       #args.train_model=0 for evaluaton


print("args.epochs",args.epochs,args.log_freq,args.random_rotate)

encoder = Encoder
decoder = MLP_Decoder  # MLP_Conv_v1  #MLP_Decoder

if torch.cuda.is_available():
    device = torch.device("cuda") 
    
model = CVAE(encoder,decoder,args)

#train model
if args.train_model == 1:
   train(model,args)

#evaliuate a trained model
if args.train_model == 0:
    args.resume_checkpoint='checkpoints/'+args.log_name+'/checkpoint-latest.pt'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(arg.gpu)
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    viz_reconstruct(model,args)
    if args.use_deterministic_encoder==False:
        sample_structure(model)






