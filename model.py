import torch.nn as nn
import torch.nn.functional as F
import torch

class LRN_ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(LRN_ResidualBlock, self).__init__()
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
        out = self.conv_block1(x)
        out = self.LRN1([out, mask, Light])
        out = self.act(out)
        out = self.conv_block2(out)
        out = self.LRN2([out, mask, Light])
        return x + out
    
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
    
    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)
        
class Lightnet(nn.Module):
    def __init__(self, in_channel=1, out_channel=128):
        super(Lightnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel//4, 1)
        self.conv2 = nn.Conv2d(out_channel//4, out_channel//2, 1)
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, mask):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        mask = F.interpolate(mask.detach(), size=out.size()[2:], mode='nearest')
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask = torch.where(mask >= 1.0, one, zero)
        Ligth = out*(1.0-mask)
        Ligth = torch.mean(Ligth, dim=[2, 3], keepdim=True)
        return Ligth

class Generator_S2F(nn.Module):
    def __init__(self):
        super(Generator_S2F, self).__init__()
        self.Light_net = Lightnet()
        self.layer0 = nn.Sequential(nn.ReflectionPad2d(3))
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 7))
        self.layer2 = nn.Sequential(LigthGuidedNormalization(32))
        self.layer3 = nn.Sequential(nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1))
        self.layer5 = nn.Sequential(LigthGuidedNormalization(64))
        self.layer6 = nn.Sequential(nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        self.layer8 = nn.Sequential(LigthGuidedNormalization(128))
        self.layer9 = nn.Sequential(nn.ReLU(inplace=True))
        self.layer10 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer11 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer12 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer13 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer14 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer15 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer16 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer17 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer18 = nn.Sequential(LRN_ResidualBlock(128))
        self.layer19 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
        self.layer20 = nn.Sequential(LigthGuidedNormalization(64))
        self.layer21 = nn.Sequential(nn.ReLU(inplace=True))
        self.layer22 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1))
        self.layer23 = nn.Sequential(LigthGuidedNormalization(32))
        self.layer24 = nn.Sequential(nn.ReLU(inplace=True))
        self.layer25 = nn.Sequential(nn.ReflectionPad2d(3))
        self.layer26 = nn.Sequential(nn.Conv2d(32, 3, 7))
        
    def forward(self, x, mask):
        Light = self.Light_net(x.detach()[:, 0, :, :].unsqueeze(1), mask)
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2([out, mask, Light])
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5([out, mask, Light])
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8([out, mask, Light])
        out = self.layer9(out)
        out = self.layer10([out, mask, Light])
        out = self.layer11([out, mask, Light])
        out = self.layer12([out, mask, Light])
        out = self.layer13([out, mask, Light])
        out = self.layer14([out, mask, Light])
        out = self.layer15([out, mask, Light])
        out = self.layer16([out, mask, Light])
        out = self.layer17([out, mask, Light])
        out = self.layer18([out, mask, Light])
        out = self.layer19(out)
        out = self.layer20([out, mask, Light])
        out = self.layer21(out)
        out = self.layer22(out)
        out = self.layer23([out, mask, Light])
        out = self.layer24(out)
        out = self.layer25(out)
        out = self.layer26(out)
        return (x+out).tanh()
        
class Generator_F2S(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator_F2S, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 32, 7), # + mask
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 32
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(32, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (self.model(x) + x).tanh() #(min=-1, max=1) #just learn a residual
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
         
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(3, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(256, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(out, out.size()[2:]).view(out.size()[0], -1).view(x.size()[0]) #global avg pool