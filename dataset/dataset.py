import os
import os.path as osp
import cv2
import json
import torch
import random
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision import transforms as T

from utils.utils import resize, crop, adjust_brightness, adjust_saturation
from utils.utils import adjust_contrast, adjust_hue

class Generation_Dataset(Dataset):
    def __init__(self, args, set_type = 'train'):
        self.args = args
        self.set_type = set_type
        self.src_modality = self.args.src_modality
        self.des_modality = self.args.des_modality
        self.data_location = self.args.data_location
        # preprocessing function of each modality
        self.scale_size = 256
        self.crop_size = 256
        self.scale_size_lowres = 128
        self.crop_size_lowres = 128
        self.preprocess = {
            'to_tensor': T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'vision': T.Compose([
                T.Resize(self.scale_size),
                T.CenterCrop(self.crop_size),
            ]),
            'touch': T.Compose([
                T.Resize((120, 160)),
                T.CenterCrop(160),
                T.Resize(self.scale_size),
                T.CenterCrop(self.crop_size),
            ]),
            'lowres': T.Compose([
                T.Resize(self.scale_size_lowres),
                T.CenterCrop(self.crop_size_lowres)
            ])
        }
        
        
        self.brightness = self.args.brightness
        self.contrast = self.args.contrast
        self.saturation = self.args.saturation
        self.hue = self.args.hue
        # load candidates
        self.cand = []
        with open(self.args.split_location) as f:
            self.cand = json.load(f)[self.set_type]  # [[obj, contact, wrong_contact]]
    
    def __len__(self):
        return len(self.cand)
    
    def colorjitter(self, srcs, brightness, contrast, saturation, hue):
        len_srcs = len(srcs)

        brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        hue_factor = np.random.uniform(-hue, hue)

        for i in range(len_srcs):
            srcs[i] = adjust_brightness(srcs[i], brightness_factor)
            srcs[i] = adjust_contrast(srcs[i], contrast_factor)
            srcs[i] = adjust_saturation(srcs[i], saturation_factor)
            srcs[i] = adjust_hue(srcs[i], hue_factor)

        return srcs
    
    def load_modality_data(self, modality, obj, contact):
        data = Image.open(
            osp.join(self.data_location, modality, str(obj), f'{contact}.png')
        ).convert('RGB')
        return data
    
    def load_modality_ref_data(self, modality, obj=1):
        if modality=='touch':
            data = Image.open(
                osp.join(self.data_location, f'{modality}_ref', f'{modality}_ref.png')
            ).convert('RGB')
        elif modality == 'vision':
            data = Image.open(
                osp.join(self.data_location, f'{modality}_ref', f'{obj}.png')
            ).convert('RGB')
        return data
    
    def __getitem__(self, index):
        obj, contact = self.cand[index]
        data = {}
        # data['names'] = [obj, contact]
        src_ref = self.load_modality_ref_data(self.src_modality, obj=obj)
        des_ref = self.load_modality_ref_data(self.des_modality, obj=obj)
        src = self.load_modality_data(self.src_modality, obj, contact)
        src_rgb = src.copy()
        des = self.load_modality_data(self.des_modality, obj, contact)
        srcs = [src_ref, src]
        srcs[0] = self.preprocess[self.src_modality](srcs[0])
        srcs[1] = self.preprocess[self.src_modality](srcs[1])
        if self.set_type == 'train':
            srcs = self.colorjitter(srcs,self.brightness,self.contrast,self.saturation,self.hue)
        srcs_lowres = []
        for i in range(len(srcs)):
            srcs_lowres += [self.preprocess['lowres'](srcs[i])]
        src_ref = srcs[0]
        src_ref_lowres = srcs_lowres[0]
        
        src = srcs[1]
        src_lowres = srcs_lowres[1]
        
        des_ref = self.preprocess[self.des_modality](des_ref)
        des_ref_lowres = self.preprocess['lowres'](des_ref)
        
        des = self.preprocess[self.des_modality](des)
        des_lowres = self.preprocess['lowres'](des)
        
        data['src_ref'] = self.preprocess['to_tensor'](src_ref)
        data['des_ref'] = self.preprocess['to_tensor'](des_ref)
        data['src_ref_lowres'] = self.preprocess['to_tensor'](src_ref_lowres)
        data['des_ref_lowres'] = self.preprocess['to_tensor'](des_ref_lowres)
        data['src'] = self.preprocess['to_tensor'](src)
        data['src_rgb'] = self.preprocess['to_tensor'](src_rgb)
        data['src_lowres'] = self.preprocess['to_tensor'](src_lowres)
        data['des'] = self.preprocess['to_tensor'](des)
        data['des_lowres'] = self.preprocess['to_tensor'](des_lowres)
        
        return data
    
    def collate(self, data):
        batch = {}
        for k in data[0].keys():
            batch[k] = torch.stack([item[k] for item in data])
        
        return batch