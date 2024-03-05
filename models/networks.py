import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from packaging import version


def define_StoF(init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    net = None
    if opt.netG == "light_blocks":
        net = LightGenerator()
    elif opt.netG == "dulight_blocks":
        net = DULightGenerator()
    elif opt.netG == "region_blocks":
        net = RegionGenerator()
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in opt.netG))

        
class DULightGenerator(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(DULightGenerator, self).__init__()
        self.DULight_net = DULightnet()
        model = [nn.ReflectionPad2d(3), 
                      nn.Conv2d(3, 32, 7), 
                      DULigthGuidedNormalization(32), 
                      nn.ReLU(inplace=True)]
        model += [nn.Conv2d(32, 64, 3, stride=2, padding=1),
                      DULigthGuidedNormalization(64), 
                      nn.ReLU(inplace=True)]
        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                       DULigthGuidedNormalization(128),
                       nn.ReLU(inplace=True),]
        for i in range(9): 
            model += [DULRN_ResidualBlock(128)]
        model += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        DULigthGuidedNormalization(64),
                        nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        DULigthGuidedNormalization(32),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(32, 3, 7)]
        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        Light, color = self.DULight_net(x.detach()[:, 0, :, :].unsqueeze(1), x.detach()[:, 1:, :, :], mask)
        out = x
        for layer in self.model:
            # import pdb; pdb.set_trace()
            if isinstance(layer, DULRN_ResidualBlock) or isinstance(layer, DULigthGuidedNormalization):
                out = layer([out, mask, Light, color])
            else:
                out = layer(out)
        return (x+out).tanh()


class RegionGenerator(nn.Module):
    def __init__(self, rate=[0.0, 0.2, 0.4, 0.8]):
        super(RegionGenerator, self).__init__()
        self.Light_net = Lightnet()
        model = [nn.ReflectionPad2d(3), 
                      nn.Conv2d(3, 32, 7), 
                      LigthGuidedNormalization(32), 
                      nn.ReLU(inplace=True)]
        model += [nn.Conv2d(32, 64, 3, stride=2, padding=1),
                      LigthGuidedNormalization(64), 
                      nn.ReLU(inplace=True)]
        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                       LigthGuidedNormalization(128),
                       nn.ReLU(inplace=True),]
        for i in range(4): 
            model += [LRN_ResidualBlock(128)]
            model += [RegionABlock(128, rate[i])]
        model += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        LigthGuidedNormalization(64),
                        nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        LigthGuidedNormalization(32),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(32, 3, 7)]
        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        Light = self.Light_net(x.detach()[:, 0, :, :].unsqueeze(1), mask)
        out = x
        for layer in self.model:
            # import pdb; pdb.set_trace()
            if isinstance(layer, LRN_ResidualBlock) or isinstance(layer, LigthGuidedNormalization):
                out = layer([out, mask, Light])
            elif isinstance(layer, RegionABlock):
                out = layer(out, mask)
            else:
                out = layer(out)
        return (x+out).tanh()

class LightGenerator(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(LightGenerator, self).__init__()
        self.Light_net = Lightnet()
        model = [nn.ReflectionPad2d(3), 
                      nn.Conv2d(3, 32, 7), 
                      LigthGuidedNormalization(32), 
                      nn.ReLU(inplace=True)]
        model += [nn.Conv2d(32, 64, 3, stride=2, padding=1),
                      LigthGuidedNormalization(64), 
                      nn.ReLU(inplace=True)]
        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                       LigthGuidedNormalization(128),
                       nn.ReLU(inplace=True),]
        for i in range(9): 
            model += [LRN_ResidualBlock(128)]
        model += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        LigthGuidedNormalization(64),
                        nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        LigthGuidedNormalization(32),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(32, 3, 7)]
        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        Light = self.Light_net(x.detach()[:, 0, :, :].unsqueeze(1), mask)
        out = x
        for layer in self.model:
            # import pdb; pdb.set_trace()
            if isinstance(layer, LRN_ResidualBlock) or isinstance(layer, LigthGuidedNormalization):
                out = layer([out, mask, Light])
            else:
                out = layer(out)
        return (x+out).tanh()


###############################################################################
# Helper Functions
###############################################################################


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class Identity(nn.Module):
    def forward(self, x):
        return x
    

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epoch) / float(opt.n_epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epoch, eta_min=1e-6)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
    

class LigthGuidedNormalization(nn.Module):
    def __init__(self, in_feature , eps=1e-5):
        super(LigthGuidedNormalization, self).__init__()
        self.eps = eps
        self.conv_gamma = nn.Sequential(nn.Conv2d(128, in_feature, 1), 
                                         nn.ReLU(inplace=False), 
                                         nn.Conv2d(in_feature, in_feature, 1))
        self.conv_beta = nn.Sequential(nn.Conv2d(128, in_feature, 1), 
                                         nn.ReLU(inplace=False), 
                                         nn.Conv2d(in_feature, in_feature, 1))
    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)


    def forward(self, input):
        x = input[0]
        mask = input[1]
        Light = input[2]
        # Light = F.interpolate(Light.detach(), size=x.size()[2:], mode='nearest')
        gamma = self.conv_gamma(Light)
        beta = self.conv_beta(Light)
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
        normalized = (x - mean_back) / std_back
        normalized_background = normalized * (1 - mask)
        
        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore 
        normalized_foreground = (normalized*gamma+beta) * mask
        
        return normalized_foreground + normalized_background
    

class Lightnet(nn.Module):
    def __init__(self, in_channel=1, out_channel=128):
        super(Lightnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel//4, 1)
        self.conv2 = nn.Conv2d(out_channel//4, out_channel//2, 1)
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x, mask):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        mask = F.interpolate(mask.detach(), size=out.size()[2:], mode='nearest')
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask = torch.where(mask >= 1.0, one, zero)
        # import pdb; pdb.set_trace()
        Ligth = out*(1.0-mask)
        # Ligth = torch.mean(Ligth, dim=[2, 3], keepdim=True)
        Ligth = self.pool(Ligth)
        return Ligth


class LRN_ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(LRN_ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_features, in_features, 3),)
        self.LRN1 = LigthGuidedNormalization(in_features)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv_block2 = nn.Sequential(nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_features, in_features, 3),)
        self.LRN2 = LigthGuidedNormalization(in_features)  


    def forward(self, input):
        x = input[0]
        mask = input[1]
        Light = input[2]
        out = self.conv_block1(x)
        out = self.LRN1([out, mask, Light])
        out = self.act(out)
        out = self.conv_block2(out)
        out = self.LRN2([out, mask, Light])
        return x + out


class DULRN_ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(DULRN_ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_features, in_features, 3),)
        self.LRN1 = LigthGuidedNormalization(in_features)
        self.act = nn.ReLU(inplace=True)
        self.conv_block2 = nn.Sequential(nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_features, in_features, 3),)
        self.LRN2 = LigthGuidedNormalization(in_features)  


    def forward(self, input):
        x = input[0]
        mask = input[1]
        Light = input[2]
        color = input[3]
        out = self.conv_block1(x)
        out = self.LRN1([out, mask, Light, color])
        out = self.act(out)
        out = self.conv_block2(out)
        out = self.LRN2([out, mask, Light, color])
        return x + out


class DULigthGuidedNormalization(nn.Module):
    def __init__(self, in_feature , eps=1e-5):
        super(DULigthGuidedNormalization, self).__init__()
        self.eps = eps
        self.conv_lgamma = nn.Sequential(nn.Conv2d(128, in_feature, 1), 
                                         nn.ReLU(inplace=False), 
                                         nn.Conv2d(in_feature, in_feature, 1))
        self.conv_lbeta = nn.Sequential(nn.Conv2d(128, in_feature, 1), 
                                         nn.ReLU(inplace=False), 
                                         nn.Conv2d(in_feature, in_feature, 1))

        self.conv_cgamma = nn.Sequential(nn.Conv2d(128, in_feature, 1), 
                                         nn.ReLU(inplace=False), 
                                         nn.Conv2d(in_feature, in_feature, 1))
        # self.conv_cbeta = nn.Sequential(nn.Conv2d(128, in_feature, 1), 
        #                                  nn.ReLU(inplace=False), 
        #                                  nn.Conv2d(in_feature, in_feature, 1))
        
    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)


    def forward(self, input):
        x = input[0]
        mask = input[1]
        Light = input[2]
        color = input[3]
        # Light = F.interpolate(Light.detach(), size=x.size()[2:], mode='nearest')
        l_gamma = self.conv_lgamma(Light)
        l_beta = self.conv_lbeta(Light)

        c_gamma = self.conv_cgamma(color)
        # c_beta = self.conv_cbeta(color)

        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
        normalized = (x - mean_back) / std_back
        normalized_background = normalized * (1 - mask)
        
        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore 
        normalized_foreground = (normalized*l_gamma+l_beta) * c_gamma * mask
        
        return normalized_foreground + normalized_background


class DULightnet(nn.Module):
    def __init__(self, l_in_channel=1, c_in_channel=2, out_channel=128):
        super(DULightnet, self).__init__()

        self.l_conv1 = nn.Conv2d(l_in_channel, out_channel//4, 1)
        self.l_conv2 = nn.Conv2d(out_channel//4, out_channel//2, 1)
        self.l_conv3 = nn.Conv2d(out_channel//2, out_channel, 1)

        self.c_conv1 = nn.Conv2d(c_in_channel, out_channel//4, 1)
        self.c_conv2 = nn.Conv2d(out_channel//4, out_channel//2, 1)
        self.c_conv3 = nn.Conv2d(out_channel//2, out_channel, 1)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, l, c, mask):

        l_out = self.act(self.l_conv1(l))
        l_out = self.act(self.l_conv2(l_out))
        l_out = self.act(self.l_conv3(l_out))

        c_out = self.act(self.c_conv1(c))
        c_out = self.act(self.c_conv2(c_out))
        c_out = self.act(self.c_conv3(c_out))

        mask = F.interpolate(mask.detach(), size=l_out.size()[2:], mode='nearest')
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask = torch.where(mask >= 1.0, one, zero)
        # import pdb; pdb.set_trace()
        Ligth = l_out*(1.0-mask)
        Ligth = torch.mean(Ligth, dim=[2, 3], keepdim=True)

        color = c_out*(1.0-mask)
        color = torch.mean(color, dim=[2, 3], keepdim=True)
        return Ligth, color
    

class RegionABlock(nn.Module):
    def __init__(self, in_channel, rate):
        super(RegionABlock, self).__init__()
        
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_channel, in_channel, 3),
                                         nn.BatchNorm2d(in_channel),
                                         nn.ReLU(inplace=True),
                                         nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_channel, in_channel, 3),
                                         nn.BatchNorm2d(in_channel),
                                         nn.ReLU(inplace=True))

        self.RA = RALayer(in_channel, rate)
    
    def forward(self, x, mask):
        # import pdb;pdb.set_trace()
        res = self.conv_block(x)
        res = self.RA(res, mask)
        res += x
        return x


class RALayer(nn.Module):
    def __init__(self, in_channel, rate, reduction=16):
        super(RALayer, self).__init__()
        self.embeding = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // reduction, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel // reduction, in_channel, 1, padding=0),
                nn.ReLU(inplace=True)
        )
        self.rate = rate

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        rate_t = torch.full_like(mask, self.rate)
        one = torch.ones_like(mask)
        mask = torch.where(mask >= 1.0, one, rate_t)
        
        y = self.embeding(x)
        y = y*mask
        return y*x
    
class colornet(nn.Module):
    def __init__(self, in_channel=2, out_channel=128):
        super(colornet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel//4, 1)
        self.conv2 = nn.Conv2d(out_channel//4, out_channel//2, 1)
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, mask):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        mask = F.interpolate(mask.detach(), size=out.size()[2:], mode='nearest')
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask = torch.where(mask >= 1.0, one, zero)
        # import pdb; pdb.set_trace()
        Ligth = out*(1.0-mask)
        # Ligth = torch.mean(Ligth, dim=[2, 3], keepdim=True)
        Ligth = self.pool(Ligth)
        return Ligth