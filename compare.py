from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import PIL#import visdomcc
#from notify import notify
import sys
import traceback
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import argparse
from main_parser import parser
import config
args = parser.parse_args()
config.init(args)
torch.manual_seed(args.seed)
from data_load import load_data
from data_load import create
import importlib
from torch.autograd import Variable
import time
import warnings#Importing
from tensorboardX import SummaryWriter
from odl.contrib import fom
import odl
#print (torch.cuda.current_device())
if(args.gpu != None):torch.cuda.set_device(args.gpu)
##
## Utilised by visual to save the images
##

##
## Shows 10 examples of images from the test set to demonstrate how well the algorithm works
## From all algorithms
##

##
## Test all algorithms and evaluates average loss, and other criteria as described in FOM
##
# def test(algorithms):
#     total_correct = 0
#     avg_loss = 0.0
#     for j, (alg, lib) in enumerate(algorithms.items()):
#         writer = SummaryWriter(config.data_path+'logs/'+alg+'test',comment=str(time.time()))
#         start=time.time()
#         all_ssim=0
#         all_psnr=0
#         for i, (scans, truths) in enumerate(data_test_loader):
#             if (scans.nelement() == 0):
#                 scans = create(truths)
#             if use_cuda:
#                 scans,truths = scans.cuda(), truths.cuda()
#             if (i % 5 == 0):
#                 print(alg+':'+str(100*i/len(data_test_loader)) + '% done')
#             reconstructions = lib.output(scans)
#             avg_loss += criterion(reconstructions, truths).detach().cpu().sum()
#             avg_ssim = np.array([fom.ssim(reconstructions[i,0].cpu(),truths[i,0].cpu()).item() for i in range(truths.shape[1])]).mean()
#             # print (odl.vector(np.array(reconstructions[i,0].cpu())).space)
#             # exit()
#             avg_psnr= np.array([fom.psnr(odl.vector(reconstructions[i,0].cpu()),odl.vector(truths[i,0].cpu())).item() for i in range(truths.shape[1])]).mean()
#             writer.add_scalar('SSIM',avg_ssim, i)
#             writer.add_scalar('PSNR',avg_psnr, i)
#             all_ssim+=avg_ssim
#             all_psnr+=avg_psnr
#         end=time.time()
#         all_ssim/=i
#         all_psnr/=i
#         avg_loss /= len(data_test_loader)/10
#         writer.add_text('Text', alg+': '+'Test Avg. Loss: %f, Accuracy: %f, Time: %f, PSNR: %f, SSIM %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test_loader),end-start, all_psnr, all_ssim), j)
#         writer.close()
#         print(alg+': '+'Test Avg. Loss: %f, Accuracy: %f, Time: %f, PSNR: %f, SSIM %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test_loader),end-start, all_psnr, all_ssim))
#


##
##Figure of Merit - PSNR and SSIM
##Algorithms to compare in form of a list, by default includes all algorithms
##
def main():#,'TV']):
    #alg_list=['TV','FBP','FBP+U','FL','TV','LPD']
    algorithms = load_nets(alg_list)
    print('Nets loaded')
    #test(algorithms)
    visual(algorithms)



import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback
from IPython.display import display, clear_output
from main_parser import parser
import config
import torchvision.transforms as transforms
from odl.contrib import fom

args = parser.parse_args()
config.init(args)
torch.manual_seed(args.seed)

from data_load import load_data, create
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim_sk
from skimage.measure import compare_psnr as psnr_sk

##
## Setting gpu settings
##
if(args.gpu != None):torch.cuda.set_device(args.gpu)


def show_image_matrix(image_batches,epoch, titles=None, indices=None, alg='', **kwargs):

    """Visualize a 2D set of images arranged in a grid.

    This function shows a 2D grid of images, where the i-th column
    shows images from the i-th batch. The typical use case is to compare
    results of different approaches with the same data, or to compare
    against a ground truth.

    Parameters
    ----------
    image_batches : sequence of `Tensor` or `Variable`
        List containing batches of images that should be displayed.
        Each tensor should have the same shape after squeezing, except
        for the batch axis.
    titles : sequence of str, optional
        Titles for the colums in the plot. By default, titles are empty.
    indices : sequence of int, optional
        Object to select the subset of the images that should be shown.
        The subsets are determined by slicing along the batch axis, i.e.,
        as ``displayed = image_batch[indices]``. The default is to show
        everything.
    kwargs :
        Further keyword arguments that are passed on to the Matplotlib
        ``imshow`` function.
    """
    if indices is None:
        displayed_batches = image_batches
    else:
        displayed_batches = [batch[indices] for batch in image_batches]

    displayed_batches = [batch.data if isinstance(batch, Variable) else batch
                         for batch in displayed_batches]


    nrows = len(displayed_batches[0])
    ncols = len(displayed_batches)
    if titles is None:
        titles = [''] * ncols

    figsize = 8
    font_size=16
    fig, rows = plt.subplots(
        nrows, ncols, sharex=True, sharey=True,
        figsize=(ncols * figsize, figsize * nrows))

    if nrows == 1:
        rows = [rows]

    for i, row in enumerate(rows):
        if ncols == 1:
            row = [row]
        for j,(name, batch, ax) in enumerate(zip(titles, displayed_batches, row)):
            if i == 0:
                ax.set_title(name)
            #plt.colorbar(cb)
#            print(batch[i].squeeze().shape)
            ax.imshow(batch[i].squeeze(), **kwargs)
            # print(batch[i].max(),batch[i].min())
            ax.set_xlabel("PSNR = {:.4f} dB, SSIM = {:.4f}, HaarPSI = {:.4f}".format(psnr_sk(displayed_batches[0][i].squeeze().cpu().numpy(),displayed_batches[j][i].squeeze().cpu().numpy(),data_range=displayed_batches[0][i].cpu().numpy().max().item()-displayed_batches[0][i].cpu().numpy().min().item()),ssim_sk(displayed_batches[0][i].squeeze().cpu().numpy(),displayed_batches[j][i].squeeze().cpu().numpy(),data_range=displayed_batches[0][i].cpu().numpy().max().item()-displayed_batches[0][i].cpu().numpy().min().item()), fom.haarpsi(displayed_batches[j][i].squeeze().cpu().numpy(),displayed_batches[0][i].squeeze().cpu().numpy())), fontsize = font_size)
            #if(j==4):ax.imshow(image_batches[3][i][0], alpha=0.5, cmap=plt.cm.coolwarm)
            # ax.set_axis_off()

    plt.savefig(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(epoch)+'.png', bbox_inches='tight')
    plt.close()

##
## Shows 10 examples of images from the validation set to demonstrate how well the algorithm works
##
def visual(alg_dict,data_loader,epoch):

    for i, (scans, truth) in enumerate(data_loader):
        if (scans.nelement() == 0):
            scans = create(truth,alg_dict['FBP'].noisemean)
        if i == 1:
            break

    test_images = Variable(truth.detach())
    test_data = Variable(scans.cuda())
    outputs=[test_images.cpu()]
    titles=['Truth']
    for alg_name,Algorithm in alg_dict.items():
        if hasattr(Algorithm,'net') and Algorithm.nograd:
            Algorithm.net.eval()
        outputs.append(Algorithm.output(test_data,test_images.cuda()).detach().cpu())
        titles.append(alg_name)
    show_image_matrix(outputs, epoch, titles, indices=slice(0, args.batch_size), clim=[0, 1],alg=args.alg,cmap='bone')

##
##Based on the number of log files calculates the experiment number
##
def cntexpir():
    if not os.path.exists(config.data_path+'logs/compare'):
        os.makedirs(config.data_path+'logs/compare')
    if (args.expir != -1):
        return args.expir
    exps = np.array([int(i[3:]) for i in (os.listdir(config.data_path+'logs/compare/'))])
    expir=1
    if (len(exps) != 0):
        expir=exps.max()+1
    print('Experiment '+str(expir))
    args.expir=expir
    return expir


def validate(alg_dict,data_loaders,expir):
    for (alg_name,Algorithm) in alg_dict.items():
        Algorithm.test()
        del Algorithm

##
## Opens all the algorithms and initialises them.
## Returns a dictionary of the algorithms in the form 'Name of algorithm': Library for the algorithm
##
def load_nets(alg_list,args,data_loaders):
    dict={}
    for alg in alg_list:
        alglib = __import__(alg)
        args.alg=alg
        dict[alg]=alglib.Algorithm(args,data_loaders,path=config.data_path+'nets/net_compare/')
        if hasattr(dict[alg],'net'):
            print(f'No of Parameters: {sum(p.numel() for p in dict[alg].net.parameters())}')
    args.alg='compare'
    return dict

def save_examples(alg_dict,data_loader,epoch):
    for i, (scans, truth) in enumerate(data_loader):
        if (scans.nelement() == 0):
            scans = create(truth,alg_dict['FBP'].noisemean)
        if i == 1:
            break

    test_images = Variable(truth.detach())
    test_data = Variable(scans.cuda())
    outputs=[test_images.cpu()]
    titles=['Truth']
    for alg_name,Algorithm in alg_dict.items():
        if hasattr(Algorithm,'net') and Algorithm.nograd:
            Algorithm.net.eval()
        outputs.append(Algorithm.output(test_data,test_images.cuda()).detach().cpu())
        titles.append(alg_name)
    for i in range(len(outputs)):
        plt.imshow((outputs[i][0,0,:,:]).numpy(),cmap='bone',vmin=0.0,vmax=1.0) #windowing: vmin=0.0,vmax=0.50
        plt.gcf().set_size_inches(5.0,5.0)
        plt.xticks([])
        plt.yticks([])
        print("{}:PSNR = {:.4f} dB, SSIM = {:.4f}".format(titles[i],psnr_sk(outputs[0][0,0,:,:].numpy(),outputs[i][0,0,:,:].numpy(),data_range=outputs[i][0,0,:,:].numpy().max().item()-outputs[i][0,0,:,:].numpy().min().item()),ssim_sk(outputs[0][0,0,:,:].numpy(),outputs[i][0,0,:,:].numpy(),data_range=outputs[i][0,0,:,:].numpy().max().item()-outputs[i][0,0,:,:].numpy().min().item())))
        plt.savefig(config.data_path+'figs/compare/'+titles[i]+'.png', bbox_inches='tight', transparent = False, pad_inches=0.1)






def main(alg_list=['FBP','FBP+U','ADR','CLAR','CLAR0','TV']):#
    print('Data loaded')
    expir = cntexpir()
    args.alg='compare'
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup))
    if not os.path.exists(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir)):
        os.makedirs(config.data_path+'figs/'+args.alg+str(args.setup)+'/'+str(args.expir))
    data_loaders = load_data(args)
    alg_dict=load_nets(alg_list,args,data_loaders)
    save_examples(alg_dict,data_loaders[1],expir)
    visual(alg_dict,data_loaders[0],expir)
    validate(alg_dict,data_loaders,expir)


if __name__ == '__main__': main()
