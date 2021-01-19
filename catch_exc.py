import numpy as np
import os
# from pyrosetta.toolbox import cleanATOM
from subprocess import Popen
import TorchProteinLibrary
import torch
data_root="/local/scratch/public/zs334/bio/data/train/"
# end_root="/local/scratch/public/zs334/bio/"
# gzip -dk file.gz
p2c = TorchProteinLibrary.FullAtomModel.PDB2CoordsUnordered()
c2tc = TorchProteinLibrary.FullAtomModel.Coords2TypedCoords()
def extract_box(coordinates):
    # print(coordinates.shape)
    coordinates=coordinates.view(-1,3)
    ##
    ## +4 and -4 correspond to the max density width, gotta change this when TorchProteinLibrary is changed
    max=torch.round(coordinates.max(dim=0)[0])+4.0
    min=torch.round(coordinates.min(dim=0)[0])-4.0
    if((max-min).max()>=180):
        return 1
    return 0

names = np.array([name for name in os.listdir(data_root)])
cnt=0
for i,name in enumerate(names):
    print(name)
    if(os.path.getsize(data_root+name)==0):
        print('Empty:',name)
        continue
    if(i%100==0):print(i)
    tr_coords_dst, tr_chain_names, tr_res_names_dst, tr_res_nums_dst, tr_atom_names_dst, tr_num_atoms_dst = p2c([data_root+name])
    coords,num_atoms_dst_tc,offsets=c2tc(tr_coords_dst,tr_res_names_dst,tr_atom_names_dst,tr_num_atoms_dst)
    cnt+=extract_box(coords)
    if(extract_box(coords)):
        print('Bad:',name)
print(cnt)
    # print(num_atoms_dst_tc.shape)
#     if(len(name.split('.'))==2):
#         Popen(['rm', data_root+name])
