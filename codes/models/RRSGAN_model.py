import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss
import torch.nn.functional as F

logger = logging.getLogger('base')

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x



class RRSGANModel(BaseModel):
    def __init__(self, opt):
        super(RRSGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.l1_init = train_opt['l1_init'] if train_opt['l1_init'] else 0

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            self.netD_grad = networks.define_D_grad(opt).to(self.device) # D_grad
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
                self.netD_grad = DistributedDataParallel(self.netD_grad,
                                                    device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
            else:
                self.netD = DataParallel(self.netD)
                self.netD_grad = DataParallel(self.netD_grad)

            self.netG.train()
            self.netD.train()
            self.netD_grad.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    pass  # do not need to use DistributedDataParallel for netF
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            # Branch_init_iters
            self.Branch_pretrain = train_opt['Branch_pretrain'] if train_opt['Branch_pretrain'] else 0
            self.Branch_init_iters = train_opt['Branch_init_iters'] if train_opt['Branch_init_iters'] else 1

            # gradient_pixel_loss
            if train_opt['gradient_pixel_weight'] > 0:
                self.cri_pix_grad = nn.MSELoss().to(self.device)
                self.l_pix_grad_w = train_opt['gradient_pixel_weight']
            else:
                self.cri_pix_grad = None

            # gradient_gan_loss
            if train_opt['gradient_gan_weight'] > 0:
                self.cri_grad_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_grad_w = train_opt['gradient_gan_weight']
            else:
                self.cri_grad_gan = None


            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # D_grad
            wd_D_grad = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D_grad = torch.optim.Adam(self.netD_grad.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))

            self.optimizers.append(self.optimizer_D_grad)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            self.get_grad = Get_gradient()
            self.get_grad_nopadding = Get_gradient_nopadding()

        self.print_network()  # print network

    def feed_data(self, data, need_GT=True):
        self.var_LQ = data['LQ'].to(self.device)  # LQ
        self.var_LQ_UX4 = data['LQ_UX4'].to(self.device)  
        self.var_Ref = data['Ref'].to(self.device)
        self.var_Ref_DUX4 = data['Ref_DUX4'].to(self.device)

        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            self.var_ref = data['GT'].clone().to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        for p in self.netD_grad.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_LQ, self.var_LQ_UX4, self.var_Ref, self.var_Ref_DUX4)

        self.fake_H_grad = self.get_grad(self.fake_H)
        self.var_H_grad = self.get_grad(self.var_H)
        self.var_ref_grad = self.get_grad(self.var_ref)
        self.var_H_grad_nopadding = self.get_grad_nopadding(self.var_H)
        self.grad_LR = self.get_grad_nopadding(self.var_LQ)

        l_g_total = 0

        if step < self.l1_init:
          l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
          l_g_pix_grad = self.l_pix_grad_w * self.cri_pix_grad(self.fake_H_grad, self.var_H_grad)
          l_g_total = l_pix + l_g_pix_grad 
          l_g_total.backward()
          self.optimizer_G.step()
          self.log_dict['l_g_pix'] = l_pix.item()
        else:
          if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            with torch.autograd.set_detect_anomaly(True):
              if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
              if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

              if self.cri_pix_grad: #gradient pixel loss
                l_g_pix_grad = self.l_pix_grad_w * self.cri_pix_grad(self.fake_H_grad, self.var_H_grad)
                l_g_total = l_g_total + l_g_pix_grad

              if self.opt['train']['gan_type'] == 'gan':
                pred_g_fake = self.netD(self.fake_H)
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
              elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                pred_g_fake = self.netD(self.fake_H)
                l_g_gan = self.l_gan_w * (
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
              l_g_total += l_g_gan

              # grad G gan + cls loss
              if self.opt['train']['gan_type'] == 'gan':
                pred_g_fake_grad = self.netD_grad(self.fake_H_grad)
                l_g_gan_grad = self.l_gan_grad_w * self.cri_gan(pred_g_fake_grad, True)
              elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real_grad = self.netD_grad(self.var_ref_grad).detach()
                pred_g_fake_grad = self.netD_grad(self.fake_H_grad)
                l_g_gan_grad = self.l_gan_grad_w * (
                    self.cri_gan(pred_d_real_grad - torch.mean(pred_g_fake_grad), False) +
                    self.cri_gan(pred_g_fake_grad - torch.mean(pred_d_real_grad), True)) / 2
              l_g_total = l_g_total + l_g_gan_grad

              l_g_total.backward()
              self.optimizer_G.step()

          # D
          for p in self.netD.parameters():
            p.requires_grad = True

          for p in self.netD_grad.parameters():
            p.requires_grad = True

          with torch.autograd.set_detect_anomaly(True):
            self.optimizer_D.zero_grad()
          # need to forward and backward separately, since batch norm statistics differ
            l_d_total = 0
            if self.opt['train']['gan_type'] == 'gan':
              pred_d_real = self.netD(self.var_ref)
              l_d_real = self.cri_gan(pred_d_real, True)
              l_d_real.backward()
              pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
              l_d_fake = self.cri_gan(pred_d_fake, False)
              l_d_fake.backward()
            elif self.opt['train']['gan_type'] == 'ragan':
              pred_d_real = self.netD(self.var_ref)
              pred_d_fake = self.netD(self.fake_H.detach())
              l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
              l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
              l_d_total = (l_d_real + l_d_fake) / 2
              l_d_total.backward()
            self.optimizer_D.step()


            self.optimizer_D_grad.zero_grad()
            l_d_total_grad = 0


            if self.opt['train']['gan_type'] == 'gan':
              pred_d_real_grad = self.netD_grad(self.var_ref_grad)
              l_d_real_grad = self.cri_grad_gan(pred_d_real_grad, True)
              l_d_real_grad.backward()
              pred_d_fake_grad = self.netD_grad(self.fake_H_grad.detach())
              l_d_fake_grad = self.cri_gan(pred_d_fake_grad, False)
              l_d_fake_grad.backward()
            elif self.opt['train']['gan_type'] == 'ragan':
              pred_d_real_grad = self.netD_grad(self.var_ref_grad)
              pred_d_fake_grad = self.netD_grad(self.fake_H_grad.detach())
              l_d_real_grad = self.cri_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True)
              pred_d_real_grad = self.netD_grad(self.var_ref_grad)
              pred_d_fake_grad = self.netD_grad(self.fake_H_grad.detach())
              l_d_fake_grad = self.cri_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False)
              l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2
              l_d_total_grad.backward()

            self.optimizer_D_grad.step()

          # set log
          if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
          # D
          self.log_dict['l_d_real'] = l_d_real.item()
          self.log_dict['l_d_fake'] = l_d_fake.item()
          # D_grad 
          self.log_dict['l_d_real_grad'] = l_d_real_grad.item()
          self.log_dict['l_d_fake_grad'] = l_d_fake_grad.item()

          # D outputs
          self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
          self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

          # D_grad outputs
          self.log_dict['D_real_grad'] = torch.mean(pred_d_real_grad.detach())
          self.log_dict['D_fake_grad'] = torch.mean(pred_d_fake_grad.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_LQ, self.var_LQ_UX4, self.var_Ref, self.var_Ref_DUX4)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])
        load_path_D_grad = self.opt['path']['pretrain_model_D_grad']
        if self.opt['is_train'] and load_path_D_grad is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D_grad))
            self.load_network(load_path_D_grad, self.netD_grad, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        self.save_network(self.netD_grad, 'D_grad', iter_step)
