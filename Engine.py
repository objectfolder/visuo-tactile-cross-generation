import os
import os.path as osp
import sys
import json
from pprint import pprint

from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision.utils as vutils
import torch.optim as optim

from utils.meters import AverageMeter
from utils.metrics import calc_psnr
from models.build import build as build_model
from dataset.build import build as build_dataset


class Engine():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        # set seeds
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        # build dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataset(
            self.args)
        # build model & optimizer
        self.model = build_model(self.args, self.cfg)
        self.model.cuda()
        # experiment dir
        self.exp_dir = osp.join('./exp_new', self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(osp.join(self.exp_dir,'viz'), exist_ok=True)
        self.visualization=False
        
    def train_epoch(self, epoch):
        epoch_info = {
            'errD': AverageMeter(),
            'errG_GAN': AverageMeter(),
            'errG_L1': AverageMeter(),
            'D_x': AverageMeter(),
            'D_G_1': AverageMeter(),
            'D_G_2': AverageMeter(),
            'PSNR': AverageMeter(),
            'SSIM': AverageMeter(),
        }
        for i, batch in tqdm(enumerate(self.train_loader), leave=False):
            output = self.model(batch, train=True)
            for epoch_info_k in epoch_info.keys():
                epoch_info[epoch_info_k].update(output[epoch_info_k],output['src'].shape[0])
            
        message = 'Train Epoch: {:03d} | Loss_D: {:.4f} | Loss_G: GAN: {:.4f} L1: {:.4f} | D(x): {:.4f} | D(G(z)): {:.4f} / {:.4f} | PSNR: {:.4f} | SSIM: {:.4f}'.format(
            epoch,
            epoch_info['errD'].avg, epoch_info['errG_GAN'].avg, epoch_info['errG_L1'].avg,
            epoch_info['D_x'].avg, epoch_info['D_G_1'].avg, epoch_info['D_G_2'].avg,
            epoch_info['PSNR'].avg, epoch_info['SSIM'].avg)
        tqdm.write(message)
        vutils.save_image(output['src'],
                        osp.join(self.exp_dir,'viz','train_epoch{:03d}_src.png').format(epoch),
                        nrow=6, normalize=True)
        vutils.save_image(output['des_lowres'],
                        osp.join(self.exp_dir,'viz','train_epoch{:03d}_des_lowres.png').format(epoch),
                        nrow=6, normalize=True)
        vutils.save_image(output['des_lowres_fake'],
                        osp.join(self.exp_dir,'viz','train_epoch{:03d}_des_lowres_fake.png').format(epoch),
                        nrow=6, normalize=True)
        
    @torch.no_grad()
    def eval_epoch(self, epoch=0, test=False):
        epoch_info = {
            'errD': AverageMeter(),
            'errG_GAN': AverageMeter(),
            'errG_L1': AverageMeter(),
            'D_x': AverageMeter(),
            'D_G_1': AverageMeter(),
            'D_G_2': AverageMeter(),
            'PSNR': AverageMeter(),
            'SSIM': AverageMeter(),
        }
        data_loader = self.test_loader if test else self.val_loader
        for i, batch in tqdm(enumerate(data_loader), leave=False):
            output = self.model(batch, train=False)
            for epoch_info_k in epoch_info.keys():
                epoch_info[epoch_info_k].update(output[epoch_info_k],output['src'].shape[0])
            if self.visualization:
                for j in range(output['src'].shape[0]):
                    print('epoch{:03d}_iter{:03d}_inst{:03d}: PSNR = {:.2f}'.format(epoch,i,j,calc_psnr(output['des_lowres'][j],output['des_lowres_fake'][j])))
                    vutils.save_image(output['src'][j],
                                    osp.join(self.exp_dir,'viz','val_epoch{:03d}_iter{:03d}_inst{:03d}_src.png').format(epoch,i,j),
                                    normalize=True)
                    vutils.save_image(output['des_lowres'][j],
                                    osp.join(self.exp_dir,'viz','val_epoch{:03d}_iter{:03d}_inst{:03d}_des_lowres.png').format(epoch,i,j),
                                    normalize=True)
                    vutils.save_image(output['des_lowres_fake'][j],
                                    osp.join(self.exp_dir,'viz','val_epoch{:03d}_iter{:03d}_inst{:03d}_des_lowres_fake.png').format(epoch,i,j),
                                    normalize=True)
        message = 'Eval Epoch: {:03d} | Loss_D: {:.4f} | Loss_G: GAN: {:.4f} L1: {:.4f} | D(x): {:.4f} | D(G(z)): {:.4f} / {:.4f} | PSNR: {:.4f} | SSIM: {:.4f}'.format(
                    epoch,
                    epoch_info['errD'].avg, epoch_info['errG_GAN'].avg, epoch_info['errG_L1'].avg, 
                    epoch_info['D_x'].avg, epoch_info['D_G_1'].avg, epoch_info['D_G_2'].avg,
                    epoch_info['PSNR'].avg, epoch_info['SSIM'].avg)
        tqdm.write(message)
        return epoch_info['PSNR'].avg, epoch_info['SSIM'].avg
    
    def train(self):
        bst_psnr = 0.0
        bst_epoch = 0
        for epoch in range(self.args.epochs):
            epoch_psnr, epoch_ssim = self.eval_epoch(epoch)
            if epoch_psnr > bst_psnr:
                print('Epoch: {} | Best PSNR reached: {:.4f}, saving best model...'.format(epoch,epoch_psnr))
                bst_psnr = epoch_psnr
                bst_epoch = epoch
                os.makedirs(osp.join(self.exp_dir,'ckpts'),exist_ok=True)
                torch.save(self.model.state_dict(), osp.join(self.exp_dir,'ckpts','bst.pth'))
            if epoch - bst_epoch >= self.args.patience:
                print('Not optimized for {} epochs, exiting training process...'.format(self.args.patience))
                break
            self.train_epoch(epoch)
    
    def test(self):
        result={}
        print("Start Testing")
        ckpt_path = osp.join(self.exp_dir,'ckpts','bst.pth')
        print("Loading best model from {}".format(ckpt_path))
        self.model.load_state_dict(torch.load(ckpt_path))
        epoch_psnr, epoch_ssim = self.eval_epoch(test = True)
        result['PSNR'] = float(epoch_psnr)
        result['SSIM'] = float(epoch_ssim)
        os.makedirs(osp.join(self.exp_dir,'result'),exist_ok=True)
        json.dump(result, open(osp.join(self.exp_dir,'result', 'result.json'),'w'))
    
    def __call__(self):
        if not self.args.eval:
            self.train()
        self.test()