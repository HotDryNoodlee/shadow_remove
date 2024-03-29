import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util


class REGIONModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.opt = opt
        self.loss_names = ['i', 'g']
        self.visual_names = ["fake_F", "real_F"]
        self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_StoF(opt=opt)

        if self.opt.isTrain:
            # import pdb;pdb.set_trace()
            self.criterionIdt = torch.nn.L1Loss().cuda()
            self.criterionRem = torch.nn.L1Loss().cuda()
            self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=0.02)
            self.optimizers.append(self.optimizer_G)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.loss = self.compute_loss()
        self.loss.backward()
        self.optimizer_G.step()


    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        # StoF = self.opt.direction == 'StoF'
        self.real_S = input['S'].cuda()
        self.real_F = input['F'].cuda()
        self.name_S = input["S_paths"]
        self.name_F = input["F_paths"]

        if self.opt.use_mask:
            self.mask = input['M'].cuda()
        self.image_paths = input['S_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.zero_mask = torch.zeros_like(self.mask, dtype=self.mask.dtype)
        # import pdb; pdb.set_trace()
        self.fake_F = self.netG(self.real_S, self.mask)
        self.idt_F = self.netG(self.real_F, self.zero_mask)
        return self.fake_F

    def compute_loss(self):
        # fake = self.fake_F.detach()
        self.loss_i = self.criterionIdt(self.real_F, self.idt_F)
        self.loss_g = self.criterionRem(self.real_F, self.fake_F)

        self.loss_G = self.loss_i + self.loss_g*10
        return self.loss_G