from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import traceback
from IPython.display import display, clear_output
import config
from torch.autograd import Variable
from TorchProteinLibrary import *
import time
import torch.nn.functional as F

##
## Custom class for loading data
##
class MyCustomDataset(Dataset):
    def __init__(self, percent, direc, transform,args):
        self.data_root = direc
        self.args=args
        self.transform = transform
        self.names = np.array([name for name in os.listdir(self.data_root)])
        if percent<0:self.names = self.names[0:-1*percent]
        else: self.names = self.names[0:int(percent*len(self.names)//100)]
        self.count = len(self.names)


    def __getitem__(self, index):
        name = self.names[index]
        #name='pdb6x6p.ent'

        # if(name!='pdb6x6p.ent'):
        print(name)
        time.sleep(0.5)
        tr_coords_dst, tr_chain_names, tr_res_names_dst, tr_res_nums_dst, tr_atom_names_dst, tr_num_atoms_dst = config.p2c([self.data_root+'/'+name])
        sequences=(config.seq_from_res_name(tr_res_names_dst,tr_atom_names_dst))
        # print(len(sequences[0]))
        coords,num_atoms_dst_tc,offsets=config.c2tc(tr_coords_dst,tr_res_names_dst,tr_atom_names_dst,tr_num_atoms_dst)
        max,min=extract_box(coords)
        # print(coords.dtype,min.dtype,tr_num_atoms_dst.dtype)
        coords=config.translate(coords.double().cpu(), -min.view(1,-1).cpu().double(), tr_num_atoms_dst.cpu())
        max,min=extract_box(coords)
        coords=coords.cuda().float()
        print(sequences[0], len(sequences[0]))

        density=(config.tc2v(coords.cuda(),num_atoms_dst_tc.cuda(),offsets.cuda())*config.weights.cuda()).squeeze().sum(dim=0,keepdim=True)
        deltas=delta_funcs(coords,num_atoms_dst_tc,offsets,density.shape,config.weights)
        max_x=int(max[0])
        max_y=int(max[1])
        max_z=int(max[2])
        density=volume_from_deltas(deltas)
        # print(max_x,max_y,max_z
        # print(density[:,:max_x,:max_y,:max_z].shape,deltas.shape)
        # exit()
        return (density[:,:max_x,:max_y,:max_z], deltas[:,:max_x,:max_y,:max_z])

    def __len__(self):
        return self.count







##
## Loading data for training/testing
##
def load_data(args,test=False):
    data_train_loader, data_valid_loader, data_test_loader = [],[],[]
    if (not test):
        data_train = MyCustomDataset(args.dataperc,args.data_path+'/train', transform=None,args=args)

        data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)#, num_workers=args.workers, drop_last=False)


        data_valid = MyCustomDataset(-args.valid,args.data_path+'/valid',transform=None,args=args)

        data_valid_loader = DataLoader(data_valid, batch_size=args.batch_size, shuffle=False)#, num_workers=args.workers, drop_last=False)
    if (test):
        data_test = MyCustomDataset(args.dataperc,args.data_path+'/test', transform=None,args=args)

        data_test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return (data_train_loader, data_valid_loader, data_test_loader)


def delta_funcs(coordinates,num_atoms,offsets,vol_shape,weights):
    deltas=torch.zeros(vol_shape).type_as(coordinates)
    coordinates=coordinates.view(-1,3)
    for i in range(coordinates.shape[0]):
        x = torch.round(config.vox_size*coordinates[i,0]).long()
        y = torch.round(config.vox_size*coordinates[i,1]).long()
        z = torch.round(config.vox_size*coordinates[i,2]).long()
        # if(deltas[0,x,y,z]!=0):
            # print('Atom clash at atom ', i)
        deltas[0, x, y, z] += weights.squeeze()[(offsets<=i).sum()-1]
    return deltas

def volume_from_deltas(deltas):
    return F.conv3d(deltas.unsqueeze(0), config.for_op,stride=1,padding=config.n//2).squeeze().unsqueeze(0)


def extract_box(coordinates):
    # print(coordinates.shape)
    coordinates=coordinates.view(-1,3)
    ##
    ## +4 and -4 correspond to the max density width, gotta change this when TorchProteinLibrary is changed
    max=torch.round(coordinates.max(dim=0)[0])+4.0
    min=torch.round(coordinates.min(dim=0)[0])-4.0
    if((max-min).max()>=180):
        print('we got a problem')
        print(max-min)
    return (max,min)
