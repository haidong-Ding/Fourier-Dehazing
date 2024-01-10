"""
paper: Pyramid Channel-based Feature Attention Network for image dehazing 
file: network.py
about: model for PCFAN
author: Tao Wang
date: 06/13/20
"""
# --- Imports --- #
from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network_5 import Net
from datasets.datasets import DehazingDataset
from os.path import exists, join, basename
import time
from torchvision import transforms
from utils import to_psnr, validation
import os


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Testing hyper-parameters for neural network')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[1], type=list)
opt = parser.parse_args()

print(opt)

# ---  hyper-parameters for testing the neural network --- #
val_batch_size = opt.testBatchSize
data_threads = opt.threads
net_path = opt.net
category = opt.category
GPUs_list = opt.n_GPUs


# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    val_data_dir = '../data/RESIDE/SOTS/indoor/nyuhaze500/'
elif category == 'outdoor':
    val_data_dir = '../data/RESIDE/SOTS/outdoor/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
test_dataset = DehazingDataset(root_dir = val_data_dir, transform = transforms.Compose([transforms.ToTensor()]), train=False)

test_dataloader = DataLoader(test_dataset, batch_size = val_batch_size, num_workers = data_threads, shuffle=False)

# --- Define the network --- #
model = Net(channels=64, num_layers=27)

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=GPUs_list)


# --- Load the network weight --- #
model.load_state_dict(torch.load('./checkpoints/6/{}_haze_300.pth'.format(category)))


# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
val_psnr, val_ssim = validation(model, test_dataloader, device, 'indoor', save_tag=False)
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))

