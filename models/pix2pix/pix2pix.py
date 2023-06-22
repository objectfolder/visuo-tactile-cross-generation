import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torchvision import models
import torchvision.utils as vutils
from torch.autograd import Variable
import math

from utils.metrics import calc_psnr, calc_ssim
from .basic_modules import init_weights, resnet18
from .losses import GANLoss

class _netD(nn.Module):
    def __init__(self, args):
        super(_netD, self).__init__()
        self.args = args
        nc = self.args.nc
        nf = self.args.ndf
        sequence = self.get_down_seq(nc * 2, nf, 1)
        self.model = nn.Sequential(*sequence)
        
    def get_down_seq(self, ni, nf, no):
        sequence = [
            # input is (ni) x 128 x 128 
            nn.Conv2d(ni, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf) x 64 x 64 
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.InstanceNorm2d(nf * 2,track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 2) x 32 x 32 
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.InstanceNorm2d(nf * 4,track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 4) x 16 x 16 
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.InstanceNorm2d(nf * 8,track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 8) x 8 x 8
            nn.Conv2d(nf * 8, nf * 16, 4, 2, 1),
            nn.InstanceNorm2d(nf * 16,track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 16) x 4 x 4 
            nn.Conv2d(nf * 16, no, 4, 1, 0),
        ]

        return sequence

    def forward(self, src, des):
        x = self.model(torch.cat([src, des], 1))
        x = x.view(x.size(0), -1)
        return x.squeeze(1)
    
class _netG(nn.Module):
    def __init__(self, args):
        super(_netG, self).__init__()
        self.args = args
        no = self.args.nc
        nf = self.args.ngf
        
        self.resnet_src = models.resnet18(True) # src
        self.resnet_src.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        
        self.num_ft = self.resnet_src.fc.in_features
        self.decoder = nn.Module()
        
        self.decoder.convt_0 = nn.ConvTranspose2d(self.num_ft, nf * 8, 8, 1, 0)
        self.decoder.norm_0 = nn.InstanceNorm2d(nf * 8,track_running_stats=True)
        
        self.decoder.convt_1 = nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)
        self.decoder.norm_1 = nn.InstanceNorm2d(nf * 4,track_running_stats=True)
        
        self.decoder.convt_2 = nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)
        self.decoder.norm_2 = nn.InstanceNorm2d(nf * 2,track_running_stats=True)
        
        self.decoder.convt_3 = nn.ConvTranspose2d(nf * 4, nf * 1, 4, 2, 1)
        self.decoder.norm_3 = nn.InstanceNorm2d(nf * 1,track_running_stats=True)
        
        self.decoder.convt_4 = nn.ConvTranspose2d(nf * 2, no, 4, 2, 1)

        init_weights(self.decoder, init_type='normal')
        # init_weights(self.resnet_src, init_type='normal')
    
    def forward_resnet(self, net, x):

        x = net.conv1(x)
        x = net.bn1(x)
        ft_0 = net.relu(x)
        x = net.maxpool(ft_0)

        ft_1 = net.layer1(x)
        ft_2 = net.layer2(ft_1)
        ft_3 = net.layer3(ft_2)
        ft_4 = net.layer4(ft_3)
        ft_5 = net.avgpool(ft_4)

        return ft_0, ft_1, ft_2, ft_3, ft_4, ft_5

    def forward(self, src):
        src_feature = self.forward_resnet(self.resnet_src, src)
        cat = src_feature[5] # (bs, 512, 1, 1)
        cat = self.decoder.convt_0(cat) # (bs, 512, 8, 8)
        cat = self.decoder.norm_0(cat) # (bs, 512, 8, 8)
        cat = F.relu(cat) # (bs, 512, 8, 8)

        cat = torch.cat((cat, src_feature[4]), 1) # (bs, 1024, 16, 16)
        cat = self.decoder.convt_1(cat) # (bs, 256, 16, 16)
        cat = self.decoder.norm_1(cat) # (bs, 256, 16, 16)
        cat = F.relu(cat) # (bs, 256, 16, 16)

        cat = torch.cat((cat, src_feature[3]), 1) # (bs, 512, 16, 16)
        cat = self.decoder.convt_2(cat) # (bs, 128, 32, 32)
        cat = self.decoder.norm_2(cat) # (bs, 128, 32, 32)
        cat = F.relu(cat) # (bs, 128, 32, 32)
        
        cat = torch.cat((cat, src_feature[2]), 1) # (bs, 256, 32, 32)
        cat = self.decoder.convt_3(cat) # (bs, 64, 64, 64)
        cat = self.decoder.norm_3(cat) # (bs, 64, 64, 64)
        cat = F.relu(cat) # (bs, 64, 64, 64)

        cat = torch.cat((cat, src_feature[1]), 1) # (bs, 128, 64, 64)
        cat = self.decoder.convt_4(cat) # (bs, 3, 128, 128)
        cat = torch.tanh(cat) # (bs, 3, 128, 128)
        
        return cat
    
class pix2pix(nn.Module):
    def __init__(self, args):
        super(pix2pix, self).__init__()
        self.args = args
        self.netG = _netG(self.args)
        self.netD = _netD(self.args)
        self.GAN_loss = GANLoss()
        self.L1_loss = nn.L1Loss()
        self.optimG = optim.AdamW(self.netG.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(args.beta1, 0.999))
        self.optimD = optim.AdamW(self.netD.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(args.beta1, 0.999))
        
    def forward(self, batch, train=True):
        output = {}
        for k in batch.keys():
            batch[k] = batch[k].cuda()
        # update netD with real
        self.netD.zero_grad()
        outD = self.netD(
            src = batch['src_lowres'],
            des = batch['des_lowres']
        )
        errD_real = self.GAN_loss(outD, True)
        D_x = outD.data.mean()
        # update netD with fakeG
        des_lowres_fake = self.netG(
            src = batch['src']
        )
        outD = self.netD(
            src = batch['src_lowres'],
            des = des_lowres_fake.detach()
        )
        errD_fake = self.GAN_loss(outD, False)
        D_G_1 = outD.data.mean()
        errD = (errD_real + errD_fake) * 0.5
        if train:
            errD.backward()
            self.optimD.step()
        
        # update netG
        self.netG.zero_grad()
        outD = self.netD(
            src = batch['src_lowres'],
            des = des_lowres_fake
        )
        D_G_2 = outD.data.mean()
        errG_GAN = self.GAN_loss(outD, True)
        errG_L1 = self.L1_loss(des_lowres_fake,
                               batch['des_lowres'])
        errG = self.args.w_GANLoss*errG_GAN + self.args.w_L1Loss*errG_L1
        if train:
            errG.backward()
            self.optimG.step()
        
        output['errD'] = errD.item()
        output['errG_GAN'] = errG_GAN.item()
        output['errG_L1'] = errG_L1.item()
        output['D_x'] = D_x.item()
        output['D_G_1'] = D_G_1.item()
        output['D_G_2'] = D_G_2.item()
        
        output['src'] = batch['src'].data
        output['des_lowres'] = batch['des_lowres'].data
        output['des_lowres_fake'] = des_lowres_fake.data
        output['PSNR'] = calc_psnr(output['des_lowres'],output['des_lowres_fake'])
        output['SSIM'] = calc_ssim(output['des_lowres'],output['des_lowres_fake'])
        
        return output