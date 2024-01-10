# --- Imports --- #
from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import Net
from datasets.datasets import DehazingDataset
from os.path import exists, join, basename
from torchvision import transforms
from utils import to_psnr, validation, print_log
import os
import time
import math


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--continueEpochs', type=int, default=0, help='continue epochs')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
parser.add_argument('--category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
opt = parser.parse_args()
print(opt)


# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = '../data/RESIDE/ITS/'
train_batch_size = opt.batchSize
val_batch_size = opt.testBatchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs
category = opt.category
continueEpochs = opt.continueEpochs


# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    val_data_dir = '../data/RESIDE/SOTS/indoor/nyuhaze500/'
elif category == 'outdoor':
    val_data_dir = '../data/RESIDE/SOTS/outdoor/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')


device_ids = GPUs_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Building model')
model = Net(channels=64, num_layers=18)


# --- Define the Loss Function --- #
L1Loss = nn.L1Loss()
L1Loss = L1Loss.to(device)


# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)

# --- Learning rate decay strategy --- #
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999)) 


# --- Load training data and validation/test data --- #
train_dataset = DehazingDataset(root_dir=train_data_dir, crop=True, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset = train_dataset, batch_size=train_batch_size, num_workers = data_threads, shuffle=True)

test_dataset = DehazingDataset(root_dir = val_data_dir, transform = transforms.Compose([transforms.ToTensor()]), train=False)

test_dataloader = DataLoader(test_dataset, batch_size = val_batch_size, num_workers = data_threads, shuffle=False)


old_val_psnr, old_val_ssim = validation(model, test_dataloader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
for epoch in range(1 + opt.continueEpochs, opt.nEpochs + 1 + opt.continueEpochs):
    print("Training...")
    start_time = time.time()
    # learning rate decay
    lr=lr_schedule_cosdecay(epoch, opt.nEpochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  
    
    psnr_list = []
    for iteration, inputs in enumerate(train_dataloader,1):

        haze, gt = Variable(inputs['hazy_image']), Variable(inputs['clear_image'])
        haze = haze.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        model.train()
        dehaze = model(haze)
        L1_loss = L1Loss(dehaze, gt)
        Loss = L1_loss

        Loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.4f} lr: {:.6f} Time: {:.2f}min".format(epoch, iteration, len(train_dataloader), Loss.item(), lr, (time.time()-start_time)/60))
        
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

    train_psnr = sum(psnr_list) / len(psnr_list)
    save_checkpoints = './checkpoints'
    if os.path.isdir(save_checkpoints)== False:
        os.mkdir(save_checkpoints)

    # --- Save the network  --- #
    torch.save(model.state_dict(), './checkpoints/{}_haze.pth'.format(category))

    # --- Use the evaluation model in testing --- #
    model.eval()

    val_psnr, val_ssim = validation(model, test_dataloader, device, category)
    
    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(model.state_dict(), './checkpoints/{}_haze_best.pth'.format(category))
        old_val_psnr = val_psnr
    
    # print_log(epoch+1, train_epoch, train_psnr, val_psnr, val_ssim, category, 'Fourier')
    