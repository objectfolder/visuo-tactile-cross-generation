import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim

def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'VisGel':
        from VisGel import VisGel
        model = VisGel.VisGel(args)
        return model
    elif args.model == 'pix2pix':
        from pix2pix import pix2pix
        model = pix2pix.pix2pix(args)
        return model