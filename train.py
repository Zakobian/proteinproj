import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback
from IPython.display import display, clear_output
from main_parser import parser
import config
import torchvision.transforms as transforms

args = parser.parse_args()
config.init(args)
torch.manual_seed(args.seed)

from data_load import load_data
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim_sk
from skimage.measure import compare_psnr as psnr_sk

##
## Setting gpu settings
##
if(args.gpu != None):torch.cuda.set_device(args.gpu)


##
## Shows 10 examples of images from the validation set to demonstrate how well the algorithm works
##
def visual(Algorithm,data_loader,epoch):
    for i, (scans, truth) in enumerate(data_loader):
        if i == 1:
            break
    Algorithm.save_img('density',scans[0])
    Algorithm.save_img('atoms',truth[0])
    Algorithm.save_img('approximation',Algorithm.output(scans,truth)[0])

##
##Based on the number of log files calculates the experiment number
##
def cntexpir():
    if not os.path.exists(config.data_path+'logs/'+args.alg):
        os.makedirs(config.data_path+'logs/'+args.alg)
    if (args.expir != -1):
        return args.expir
    exps = np.array([int(i[3:]) for i in (os.listdir(config.data_path+'logs/'+args.alg+'/'))])
    expir=1
    if (len(exps) != 0):
        expir=exps.max()+1
    print('Experiment '+str(expir))
    args.expir=expir
    return expir


def train_and_test(epoch,Algorithm,expir,data_loaders):
    #Algorithm.print_weights()
    if(not args.visual):
        writer = SummaryWriter(config.data_path+'logs/'+args.alg+'/exp'+str(expir)+'/train/'+str(epoch),comment='')
        Algorithm.train(writer,epoch)
        with SummaryWriter(config.data_path+'logs/'+args.alg+'/exp'+str(expir)+'/hypers',comment='Hype') as hypw:
            Algorithm.validate(writer,hypw,epoch)
        writer.close()
    visual(Algorithm,data_loaders[0],epoch)
    if(args.visual): exit()

def main():
    data_loaders = load_data(args)
    print('Data loaded')

    expir = cntexpir()
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup))
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir))


    alglib = __import__(args.alg)
    Algorithm=alglib.Algorithm(args,data_loaders)

    print(args.alg + ' net loaded in')

    with SummaryWriter(config.data_path+'logs/'+args.alg+'/exp'+str(expir)+'/',comment='') as w:
        w.add_text('Seed',str(args.seed))
        w.add_text('Setup',str(args.setup))
        w.add_text('Learning Rate',str(args.lr))
        w.add_text('Parameters',str(args))
        w.add_text('Data Percentage',str(args.dataperc))
        if hasattr(Algorithm,'net'):
            w.add_text('No of Parameters',str(sum(p.numel() for p in Algorithm.net.parameters())))
        if hasattr(Algorithm,'phi1net'):
            w.add_text('No of Parameters',str(sum(p.numel() for p in Algorithm.phi1net.parameters())))


    for ep in range(1, args.epochs+1):
        train_and_test(ep,Algorithm,expir,data_loaders)

    #Algorithm.validate(w, end_res=True)

if __name__ == '__main__': main()
