import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import sys
import config
import os
import torch.nn.functional as F
import time
from torchvision import utils
import skimage
from skimage.measure import compare_ssim as ssim_sk
from skimage.measure import compare_psnr as psnr_sk
import mrcfile


class baseNet():
    def __init__(self,args,path,net,data_loaders):
        self.args=args
        self.alg=args.alg
        self.net=net
        if(args.mult):path += 'net'+str(args.alg)+str(config.setup)+'-'+str(args.expir)+'.pt'
        else:path += 'net'+str(args.alg)+str(config.setup)+'.pt'
        self.load_net(path)
        self.optimizer=torch.optim.Adam(net.parameters(),lr=self.args.lr)
        self.data_train_loader, self.data_valid_loader, self.data_test_loader = data_loaders
        self.hypers={'Seed':args.seed,'Noise':int(args.noise*100),'Data Percentage':args.dataperc}
        self.nograd=True
        print(f'No of Parameters: {sum(p.numel() for p in self.net.parameters())}')
        if(self.args.cuda):net.cuda()

    def change_lr(self,new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr

    def load_net(self,name):
        # print(name)
        if (os.path.isfile(name) and self.args.load):
            self.net.load_state_dict(torch.load(name))
            print('Loaded {} from checkpoint'.format(self.args.alg))
        elif self.args.init: pass#self.net.apply(self.net.init_weights)

    def save_checkpoint(self, name):
        torch.save(self.net.state_dict(), name)

    def loss(self, scans, truth):
        return nn.MSELoss()(self.output(scans),truth)

    def save_img(self,name,img):
        with torch.no_grad():
            with mrcfile.new(config.data_path+'figs/'+self.args.alg+str(self.args.setup)+'/'+str(self.args.expir)+'/'+name+'.mrc') as mrc:
                mrc.set_data(img.cpu().numpy().squeeze())


    def train(self,writer,epoch):
        if hasattr(self,'net'):self.net.train()
        for i, (scans, truth) in enumerate(self.data_train_loader):
            start = time.time()

            loss = self.train_one_batch(scans,truth,writer)

            end = time.time()

            writer.add_scalar('Loss', loss, (epoch-1)*len(self.data_train_loader)+i)
            if i % self.args.log_interval == 0:
                print(str(self.args.expir) + ':Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Time: {:.6f}s'.format(epoch, i * len(scans), len(self.data_train_loader.dataset),100. * i / len(self.data_train_loader), loss,end-start))
                with open('./logs/'+self.args.alg+str(self.args.expir)+'.txt',"a") as f:
                    f.write('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Time: {:.6f}s\n'.format(epoch, i * len(scans), len(self.data_train_loader.dataset),100. * i / len(self.data_train_loader), loss,end-start))
            if i % self.args.cp_interval == 0:
                self.save_checkpoint(config.data_path+'nets/net_cps/net'+self.args.alg+str(self.args.setup)+'ep'+str(epoch)+'no'+str(i)+'exp'+str(self.args.expir)+'.pt')

    def train_one_batch(self,scans,truth,writer):
        if self.args.cuda:
            scans,truth = scans.cuda(), truth.cuda()
        self.optimizer.zero_grad()
        loss = self.loss(scans, truth)
        loss.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        return loss.item()

    def validate(self,writer,hyp_writer,epoch,end_res=False,test=False):
        if hasattr(self,'net'): self.net.eval()
        if(self.nograd):
            with torch.no_grad(): self.validate_cycle(epoch,end_res,test,writer,hyp_writer)
        else: self.validate_cycle(epoch,end_res,test,writer,hyp_writer)

    def validate_cycle(self,epoch,end_res,test,writer=None,hyp_writer=None):
        avg_loss=0.0
        all_ssim=0.0
        all_psnr=0.0
        avg_mse_loss=0.0
        print("Validate cycle of {}".format(self.alg))
        if test: data=self.data_test_loader
        else: data=self.data_valid_loader

        start=time.time()
        for i, (scans, truth) in enumerate(data):
            if self.args.cuda:
                scans,truth = scans.cuda(), truth.cuda()
            output = self.output(scans,truth)
            cur_loss = self.loss(scans, truth)
            if (type(cur_loss) is tuple): #Checking that we only have one loss not Multiple
                cur_loss=cur_loss[0]
            cur_loss=cur_loss.detach().cpu().item()
            mse_loss = nn.MSELoss()(output,truth).detach().cpu().item()

            avg_ssim = self.ssim(output,truth)
            avg_psnr = self.psnr(output,truth)

            avg_loss+= cur_loss
            avg_mse_loss+=mse_loss
            all_ssim+=avg_ssim
            all_psnr+=avg_psnr
            if (writer is not None):
                writer.add_scalar('SSIM',avg_ssim,i, (epoch-1)*len(data)+i)
                writer.add_scalar('PSNR',avg_psnr,i, (epoch-1)*len(data)+i)
                writer.add_scalar('Validation Loss',cur_loss,i, (epoch-1)*len(data)+i)
                writer.add_scalar('MSE Loss', mse_loss,i, (epoch-1)*len(data)+i)
        end=time.time()
        avg_loss/=len(data)
        avg_mse_loss/=len(data)
        all_ssim/=len(data)
        all_psnr/=len(data)
        timed=(end-start)/len(data)

        if (writer is not None): writer.add_text('Text','Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %f, SSIM %f' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))
        # if (hyp_writer is not None):
        #     met_dict = {'hparam/SSIM':all_ssim,
        #             'hparam/PSNR':all_psnr,
        #             'hparam/Loss':avg_loss,
        #             'hparam/MSELoss':avg_mse_loss}
        #
        #     hyp_writer.add_hparams(self.hypers,met_dict,epoch)
        with open('./logs/'+self.args.alg+str(self.args.expir)+'.txt',"a") as f:
            f.write('Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %f, SSIM %f\n' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))
        print('Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %f, SSIM %f' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))

    def test(self):
        self.validate(end_res=True,test=False,writer=None,hyp_writer=None,epoch=1)

    def ssim(self,output,truth):
        avssim = 0
        for i in range(truth.shape[0]):
            for j in range(truth.shape[1]):
                avssim += ssim_sk(truth[i,j].cpu().detach().numpy(),output[i,j].cpu().detach().numpy(),data_range=truth[i,j].max().detach().item()-truth[i,j].min().detach().item())
        return avssim/(truth.shape[0]*truth.shape[1])

    def psnr(self,output,truth):
        avpsnr = 0
        for i in range(truth.shape[0]):
            for j in range(truth.shape[1]):
                avpsnr += psnr_sk(truth[i,j].cpu().detach().numpy(),output[i,j].cpu().detach().numpy(),data_range=truth[i,j].max().detach().item()-truth[i,j].min().detach().item())
        return avpsnr/(truth.shape[0]*truth.shape[1])


    def output(self,scans,truth=None):
        return self.net(scans)



class baseOpt():
    def __init__(self,args,data_loaders):
        self.alg=args.alg
        self.eps=args.eps
        self.args=args
        self.iterates=args.iterates
        self.data_train_loader, self.data_valid_loader, self.data_test_loader = data_loaders
        self.nograd=False
        # print(self.data_train_loader, self.data_valid_loader, self.data_test_loader)

    def ssim(self,output,truth):
        avssim = 0
        for i in range(truth.shape[0]):
            for j in range(truth.shape[1]):
                avssim += ssim_sk(truth[i,j].cpu().detach().numpy(),output[i,j].cpu().detach().numpy(),data_range=truth[i,j].max().detach().item()-truth[i,j].min().detach().item())
        return avssim/(truth.shape[0]*truth.shape[1])

    def psnr(self,output,truth):
        avpsnr = 0
        for i in range(truth.shape[0]):
            for j in range(truth.shape[1]):
                avpsnr += psnr_sk(truth[i,j].cpu().detach().numpy(),output[i,j].cpu().detach().numpy(),data_range=truth[i,j].max().detach().item()-truth[i,j].min().detach().item())
        return avpsnr/(truth.shape[0]*truth.shape[1])
    def save_img(self,name,img):
        with torch.no_grad():
            with mrcfile.new(config.data_path+'figs/'+self.args.alg+str(self.args.setup)+'/'+str(self.args.expir)+'/'+name+'.mrc') as mrc:
                mrc.set_data(img.cpu().numpy().squeeze())


    def output(self,scans,truth=None):
        pass
    def validate(self,writer,hyp_writer,epoch,end_res=False,test=False):
        print(test)
        if hasattr(self,'net'): self.net.eval()
        if(self.nograd):
            with torch.no_grad(): self.validate_cycle(epoch,end_res,test,writer,hyp_writer)
        else: self.validate_cycle(epoch,end_res,test,writer,hyp_writer)

    def validate_cycle(self,epoch,end_res,test,writer=None,hyp_writer=None):
        avg_loss=0.0
        all_ssim=0.0
        all_psnr=0.0
        avg_mse_loss=0.0
        print("Validate cycle of {}".format(self.alg))
        if test: data=self.data_test_loader
        else: data=self.data_valid_loader
        # print(test)
        # print(data)
        start=time.time()
        for i, (scans, truth) in enumerate(data):
            if (scans.nelement() == 0):
                scans = create(truth,self.noisemean)
            if self.args.cuda:
                scans,truth = scans.cuda(), truth.cuda()
            output = self.output(scans,truth)
            cur_loss = self.loss(scans, truth)
            if (type(cur_loss) is tuple): #Checking that we only have one loss not Multiple
                cur_loss=cur_loss[0]
            cur_loss=cur_loss.detach().cpu().item()
            mse_loss = nn.MSELoss()(output,truth).detach().cpu().item()

            avg_ssim = self.ssim(output,truth)
            avg_psnr = self.psnr(output,truth)

            avg_loss+= cur_loss
            avg_mse_loss+=mse_loss
            all_ssim+=avg_ssim
            all_psnr+=avg_psnr
            if (writer is not None):
                writer.add_scalar('SSIM',avg_ssim,i, (epoch-1)*len(data)+i)
                writer.add_scalar('PSNR',avg_psnr,i, (epoch-1)*len(data)+i)
                writer.add_scalar('Validation Loss',cur_loss,i, (epoch-1)*len(data)+i)
                writer.add_scalar('MSE Loss', mse_loss,i, (epoch-1)*len(data)+i)
        end=time.time()
        avg_loss/=len(data)
        avg_mse_loss/=len(data)
        all_ssim/=len(data)
        all_psnr/=len(data)
        timed=(end-start)/len(data)

        if (writer is not None): writer.add_text('Text','Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %f, SSIM %f' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))
        if (hyp_writer is not None):
            met_dict = {'hparam/SSIM':all_ssim,
                    'hparam/PSNR':all_psnr,
                    'hparam/Loss':avg_loss,
                    'hparam/MSELoss':avg_mse_loss}

            hyp_writer.add_hparams(self.hypers,met_dict,epoch)
        with open('./logs/'+self.args.alg+str(self.args.expir)+'.txt',"a") as f:
            f.write('Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %f, SSIM %f\n' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))
        print('Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %f, SSIM %f' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))

    def test(self):
        self.validate(end_res=True,test=False,writer=None,hyp_writer=None,epoch=1)

    def loss(self,scans,truth):
        return torch.tensor(0.0)
