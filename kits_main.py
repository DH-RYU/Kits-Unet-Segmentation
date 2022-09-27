import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import torch
from torch import nn
import torchvision.transforms as transforms

from starter_code.utils import load_case,load_volume
from starter_code.visualize import visualize
from starter_code.evaluation import evaluate
from kits_solver import Solver

import argparse
from tqdm.auto import tqdm


def main(config):
    solver = Solver(config)
    solver.build_model()
    if config.mode == 'train' :
        solver.train()
    elif config.mode == 'test' :
        solver.test()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # Model configuration.
    parser.add_argument('--input_dim',type = int,default=1,help='input dimension')
    parser.add_argument('--output_dim',type = int,default=3,help='output dimension')
    parser.add_argument('--num_layers', type=int, default=5, help='numbers of u_net blocks')
    parser.add_argument('--residual_path', type=bool, default=False, help='decide whether use residual path or not')
    parser.add_argument('--resize',type=bool,default=False,help='resize')
    parser.add_argument('--random_crop',type=bool,default=True,help='random_crop')

    parser.add_argument('--train_load', type=bool, default=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    parser.add_argument('--lr',type=float,default=1e-04)
    parser.add_argument('--n_epochs',type=int,default=500)
    parser.add_argument('--eps',type=float,default=1e-08)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--device',type=str,default="cuda:0" if torch.cuda.is_available else "cpu")

    parser.add_argument('--mode', type=str, default='train', choices=['train','val', 'test'])
    parser.add_argument('--load_epoch', type=int, default=81,help='load_peoch for testing')

    parser.add_argument('--kits_directory',type=str, default=os.path.join(os.getcwd(),'kits_data'))
    
    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(),'saved_model','kits'))
    parser.add_argument('--save_name',type=str,default='kits_unet_2',help='Saving name')
    config = parser.parse_args()
    main(config)