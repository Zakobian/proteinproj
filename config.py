import torch.nn as nn
import torch
import numpy as np
import sys, termios, tty, os, time
import fcntl
from TorchProteinLibrary import *
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")##Ignore warnings



##
## Function for receiving input from keyboard
##
def getch():
  fd = sys.stdin.fileno()

  oldterm = termios.tcgetattr(fd)
  newattr = termios.tcgetattr(fd)
  newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
  termios.tcsetattr(fd, termios.TCSANOW, newattr)

  oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
  fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

  try:
    while 1:
      try:
        c = sys.stdin.read(1)
        break
      except IOError: pass
  finally:
    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
  return c



##
## Making sure that algortihms can be loaded from the Algorithms folder
##
sys.path.append('./Algorithms/')





def seq_from_res_name(res_names_dst,atom_names_dst):
    is0C = torch.eq(atom_names_dst[:,:,0], 67).squeeze()
    is1A = torch.eq(atom_names_dst[:,:,1], 65).squeeze()
    is20 = torch.eq(atom_names_dst[:,:,2], 0).squeeze()
    isCA = is0C*is1A*is20
    out=[]
    out_str=''
    dicti={ 'GLY':'G',
            'ALA':'A',
            'LEU':'L',
            'MET':'M',
            'PHE':'F',
            'TRP':'W',
            'LYS':'K',
            'GLN':'Q',
            'SER':'S',
            'PRO':'P',
            'VAL':'V',
            'ILE':'I',
            'CYS':'C',
            'TYR':'Y',
            'HIS':'H',
            'ARG':'R',
            'ASN':'N',
            'ASP':'D',
            'GLU':'E',
            'THR':'T'}
    for i,strin in enumerate(res_names_dst.squeeze().numpy()):
        out_str=''
        for char in (strin):
            if char!=0:
                out_str+=(chr(char))
#         print(out_str)

        if isCA[i]:
            if(out_str in dicti):
                out_str=dicti[out_str]

            out.append(out_str)
#         if(i==109):
#             print(out_str)
#             break
#         print(i)
    return [''.join(out)]







##
## Initialise opertaors and the data parameters
##
def init(args):
    # global size, angles, noise, rec_size,setup,detectors
    global data_path,weights,setup,for_op,n,vox_size
    global a2c,rmsd,vc,tc2v,c2c,translate,rotate,p2c,c2tc

    noise = args.noise
    setup = args.setup
    box=180
    # weights=torch.tensor([16.0,7.0,7.0,7.0,7.0,8.0,8.0,8.0,6.0,6.0,6.0]).view(1,-1,1,1,1)
    weights=torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).view(1,-1,1,1,1)


    a2c = FullAtomModel.Angles2Coords()
    rmsd = RMSD.Coords2RMSD()
    vc=Volume.VolumeConvolution()
    tc2v=Volume.TypedCoords2Volume(box_size=box)

    #Coords transforms
    c2c = FullAtomModel.Coords2Center()
    translate = FullAtomModel.CoordsTranslate()
    rotate = FullAtomModel.CoordsRotate()

    p2c = FullAtomModel.PDB2CoordsUnordered()
    c2tc = FullAtomModel.Coords2TypedCoords()


    ##
    ## clean pdb files
    ##
    data_path = args.data_path

    if(args.setup == 1):
        args.data_path = args.data_path +'/data'

    vox_size=1
    n=11
    gauss_real_width = args.res/np.pi
    # x = -i * table_step_size * apix / gauss_real_width;
    # if(addpdbbfactor==-1){
    # table[i] = exp(-x * x);
    # sigma=args.res/(np.pi*vox_size)
    # print('Sigma:',sigma)
    for_op = torch.zeros(n,n,n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                center=n//2
                r2=(center-i)**2+(center-j)**2+(center-k)**2
                for_op[i,j,k]=np.exp(-r2*(0.001 * vox_size / gauss_real_width)**2)
    for_op=for_op.view(1,1,n,n,n).cuda()
