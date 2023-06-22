import argparse
import yaml
from easydict import EasyDict as edict
from Engine import Engine

def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_objects", type=int, default=300)
    parser.add_argument("--src_modality", type=str, default='touch')
    parser.add_argument("--des_modality", type=str, default='vision')
    parser.add_argument("--model", type=str, default="VisGel")
    parser.add_argument("--pretrain", type=str, default='None')
    parser.add_argument("--config_location", type=str, default="./configs/default.yml")
    parser.add_argument('--eval', action='store_true', default=False, help='if True, only perform testing')
    # Data Locations
    parser.add_argument("--data_location", type=str, default='../DATA_new')
    parser.add_argument("--split_location", type=str, default='../DATA_new/split.json')
    # Train & Evaluation
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=200)
    # Exp
    parser.add_argument("--exp", type=str, default='test', help = 'The directory to save checkpoints and results')
    # Data Preprocess
    parser.add_argument('--brightness', type=float, default=0.3)
    parser.add_argument('--contrast', type=float, default=0.3)
    parser.add_argument('--saturation', type=float, default=0.3)
    parser.add_argument('--hue', type=float, default=0.2)
    # Model Config
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--np', type=int, default=7)
    parser.add_argument('--w_L1Loss', type=float, default=10.0)
    parser.add_argument('--w_GANLoss', type=float, default=1.0)
    
    args = parser.parse_args()
    return args

def get_config(args):
    cfg_path = args.config_location
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)

def main():
    args = parse_args()
    cfg = get_config(args)
    engine = Engine(args, cfg)
    engine()
    
if __name__ == "__main__":
    main()