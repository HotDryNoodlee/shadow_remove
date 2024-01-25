import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util


class GANmodel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        self.opt = opt
        self.loss_names = ["D_GAN", "G_GAN", "I", "D_real", "D_fake"]
        self.model_names = ['G', 'D']
        self.visual_names = ["fake_F", "real_F"]
        self.optimizers = []
        self.netG = networks.define_StoF(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, opt.gpu_ids, opt)
        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, opt.gpu_ids, opt)
        self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionIdt = torch.nn.L1Loss().to(self.device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        StoF = self.opt.direction == 'StoF'
        self.real_S = input['S' if StoF else 'F'].to(self.device)
        self.real_F = input['F' if StoF else 'S'].to(self.device)
        self.image_paths = input['S_paths' if StoF else 'F_paths']


    def forward(self):

        self.fake_F = self.netG(self.real_S)
        self.idt_F = self.netG(self.real_F)
        return self.fake_F

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def compute_D_loss(self):

        fake = self.fake_F.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        self.pred_real = self.netD(self.real_F)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        self.loss_D_GAN = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D_GAN


    def compute_G_loss(self):

        self.loss_I = self.criterionIdt(self.real_F, self.idt_F)

        fake = self.fake_F
        pred_fake = self.netD(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, False).mean()

        return self.loss_I+self.loss_G_GAN

