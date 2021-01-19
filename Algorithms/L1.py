# import odl
import torch
import config
import numpy as np
import BaseAlg
import torch.nn.functional as F

# from data_load import volume_from_deltas

def volume_from_deltas(deltas):
    return F.conv3d(deltas, config.for_op,stride=1,padding=config.n//2)



class Algorithm(BaseAlg.baseOpt):
    def __init__(self,args,data_loaders):
        super(Algorithm, self).__init__(args,data_loaders)
        self.cntr=1


    def output(self,density,truth=None,lambd=0):
        eps=self.eps
        # grad = torch.zeros(guess.shape).type_as(guess)
        # noise=torch.randn(guess.shape).type_as(guess)/10
        noise=torch.zeros(density.shape).type_as(density).normal_(0, self.args.mu)
        noise_norm=torch.norm(noise.detach(),p=2)
        density+=noise
        guess=torch.zeros(density.shape).type_as(density)
        guess=torch.nn.Parameter(guess)
        optimizer = torch.optim.Adam([guess], lr=self.eps)
        #scheduler = StepLR(optimizer, step_size=100, gamma=1.15)

        #print('Mean noise output',self.net(noise).mean())
        #print(guess.max(),guess.min())
        print(density.shape)
        # print(truth.shape)
        lamda=1e-5
        self.cntr+=1
        for j in range(self.args.iterates):
            # grad_net = self.lamb * self.grady(guess)
            # guess -= eps*(grad_net+grad)
            # if(j>600): eps=0.95*eps

            c = config.getch()
            if(c =='q'):
                break
            #scheduler.step()
            with torch.no_grad():guess.clamp_(0)
            optimizer.zero_grad()
            lossm=torch.norm(volume_from_deltas(guess)-density,p=2)+self.args.alpha*torch.norm(guess,p=1)
            if(truth is not None and j%self.args.log_interval==0):
                loss = torch.norm(guess.detach()-truth.detach().cuda(),p=2)
                cur_loss = 0#self.net(guess.detach()).mean()
                if(self.args.outp):
                    print(j)
                    print('MSE Loss:', loss.item())
                    print('Loss:',lossm.item())
                    print('Noise:',noise_norm.item())
                    print('Total:',guess.sum().item(),'/',truth.sum().item())
            lossm.backward()

            optimizer.step()
        return guess
    # #TV optimizer in ODL
    # def tv_optimizer(x_init, x_ground_truth, noisy_data_np, n_iter, lambda_tv, space=space, solver='pdhg'):
    #     #x_init, x_ground_truth, noisy_data_np must be numpy arrays
    #     x_true_odl = space.element(x_ground_truth)
    #     noisy_data_odl = ray_transform_operator.range.element(noisy_data_np)
    #     x_true_np = x_true_odl.asarray()
    #     grad_odl = odl.Gradient(space)
    #     op_odl = odl.BroadcastOperator(ray_transform_operator, grad_odl)
    #     #init
    #     x_init_odl = space.element(x_init)
    #     #setup the problem in ODL
    #     f_odl = odl.solvers.ZeroFunctional(op_odl.domain)
    #     l2_norm_squared = odl.solvers.L2NormSquared(ray_transform_operator.range).translated(noisy_data_odl) #data must be in ODL format
    #     l1_norm = lambda_tv * odl.solvers.L1Norm(grad_odl.range)
    #     g_odl = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)
    #     op_norm_odl = 1.1 * odl.power_method_opnorm(op_odl)
    #     if(solver=='pdhg'): #PDHG solver (https://github.com/odlgroup/odl/blob/master/examples/solvers/pdhg_tomography.py)
    #         tau_odl = 1.0 / op_norm_odl
    #         sigma_odl = 1.0 / op_norm_odl
    #         print('step-sizes in PDHG: sigma = {:.6f}, tau = {:.6f}'.format(sigma_odl, tau_odl))
    #         odl.solvers.nonsmooth.primal_dual_hybrid_gradient.pdhg(x_init_odl, f_odl, g_odl, op_odl, niter=n_iter, tau=tau_odl, sigma=sigma_odl)
    #     else: #ADMM solver (https://github.com/odlgroup/odl/blob/master/examples/solvers/admm_tomography.py)
    #         sigma_odl = 2.0
    #         tau_odl = sigma_odl / (op_norm_odl ** 2)
    #         print('step-sizes in ADMM: sigma = {:.6f}, tau = {:.6f}'.format(sigma_odl, tau_odl))
    #         odl.solvers.admm_linearized(x_init_odl, f_odl, g_odl, op_odl, tau=tau_odl, sigma=sigma_odl, niter=n_iter)
    #     #clamp the output to [0,1]
    #     x_tv_odl_np = x_init_odl.asarray()
    #     x_tv_odl_np = cut_image(x_tv_odl_np, vmin=0.0, vmax=1.0)
    #     #evaluate the loss
    #     x_tv_odl_torch = torch.from_numpy(x_tv_odl_np).view(1,1,img_size,img_size).to(device)
    #     noisy_data_torch = torch.from_numpy(noisy_data_np).view(1,1,num_angles,num_detector_pixels).to(device)
    #     data_loss_odl = sq_loss(noisy_data_torch, op(x_tv_odl_torch))
    #     prior_odl = tv_reg(x_tv_odl_torch, lambda_tv)
    #     variational_loss_odl =  data_loss_odl + prior_odl
    #     print('TV loss ODL = {:.6f}, data loss ODL = {:.6f}, prior ODL = {:.6f}'.\
    #           format(variational_loss_odl,data_loss_odl,prior_odl))
    #     return x_tv_odl_np
