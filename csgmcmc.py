'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.autograd import Variable
import numpy as np
import random

import sklearn
from whitebox_attack import *
from blackbox_attack import *
import argparse
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from user import *
from data import *
from tqdm import tqdm
import copy
from fed_attack import *
from opacus import PrivacyEngine
from model_utils import _ECELoss
from model_utils import _batchnorm_to_groupnorm_new
from diffmi_attack import diffmi_attack
from nasr_fed_attack import nasr_fed_attack
from multi_party_attack import *

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=int, default=1,
                    help='1: SGLD')
parser.add_argument('--device_id',type = int, help = 'device id to use',default=0)
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1./50000,
                    help='temperature (default: 1/dataset_size)')
parser.add_argument('--dataset',type=str,default='cifar10')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
sklearn.utils.check_random_state(args.seed)
#dataset = args.dataset
#model_name = 'alexnet'
#model_name = 'resnet20'
#model_name = 'intel'
#model_name = 'retina'
#model_name = 'sat6'
model_name = 'resnet18'
#model_name = 'fashion_mnist'
target_dataset = dataset(dataset_name=args.dataset, gpu=1,membership_attack_number=0)
datasize = 45000
#datasize = 10000
#datasize = 20000
#datasize = 20000
#datasize = 20000
num_batch = datasize/args.batch_size+1
lr_0 = 0.5 # initial lr for resnet20 / resnet18
#lr_0 = 0.02# initlal lr for intel
#lr_0 = 0.2 # initial lr for retina
#lr_0 = 0.1 # intial lr for sat6
#lr_0 = 0.2 # initlal lr for fashion_mnist
M = 4 # number of cycles
#M = 3
T = args.epochs*num_batch # total number of iterations
criterion = nn.CrossEntropyLoss()
#net = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
net = ResNet18(10)
#net = resnet(depth=20,num_classes=10)
optimizer = optim.SGD(net.parameters(), lr=lr_0, momentum=1-args.alpha, weight_decay=5e-4)
mt = 0

# Model
print('==> Building model..')
#net = alexnet(num_classes=10)
#net = resnet(depth=20,num_classes=10)
#net = ResNet18(num_classes=10)

if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True


def assign_part_dataset(dataset):
	if (dataset.dataset_name == 'cifar10' or dataset.dataset_name == 'cifar100'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),  ### totensor will perform the divide by 255 op
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'mnist' or dataset.dataset_name == 'fashion_mnist' or (
			'celeb' in dataset.dataset_name) or dataset.dataset_name == 'sat6' or dataset.dataset_name == 'retina' or (
			'fashion_product' in dataset.dataset_name) or dataset.dataset_name == 'intel' or dataset.dataset_name == 'gtsrb'):
		transform_train = transforms.ToTensor()
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	train_data = np.load(f'./csgmcmc/{args.dataset}_{int(datasize/10)}_train_data.npy')
	train_label = np.load(f'./csgmcmc/{args.dataset}_{int(datasize/10)}_train_label.npy')
	test_data = np.load(f'./csgmcmc/{args.dataset}_{int(datasize/10)}_test_data.npy')
	test_label = np.load(f'./csgmcmc/{args.dataset}_{int(datasize/10)}_test_label.npy')
	
	print (train_data.shape,test_data.shape,train_label.shape,test_label.shape)
	
	#### create dataset and dataloader
	train = part_pytorch_dataset(train_data, train_label, train=True, transform=transform_train,
	                             target_transform=target_transform)
	test = part_pytorch_dataset(test_data, test_label, train=False, transform=transform_test,
	                            target_transform=target_transform)
	
	train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=1)
	test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=1)
	
	# remaining = np.setdiff1d(np.arange(len(train_label)),assigned_index)
	# print (remaining[:20])
	# print (np.sort(assigned_index)[:20])
	
	return train_data_loader, test_data_loader

trainloader,testloader = assign_part_dataset(target_dataset)

def noise_loss(lr,alpha):
    noise_loss = 0.0
    noise_std = (2/lr*alpha)**0.5
    for var in net.parameters():
        means = torch.zeros(var.size()).cuda(device_id)
        noise_loss += torch.sum(var * torch.normal(means, std = noise_std).cuda(device_id))
    return noise_loss

def adjust_learning_rate(optimizer, epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets,_) in enumerate(trainloader):
		if use_cuda:
			inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
		
		optimizer.zero_grad()
		lr = adjust_learning_rate(optimizer, epoch,batch_idx)
		#print (f"epoch {epoch}, lr {lr}")
		outputs = net(inputs)
		if (epoch%50)+1>45:
			loss_noise = noise_loss(lr,args.alpha)*(args.temperature/datasize)**.5
			loss = criterion(outputs, targets)+loss_noise
		else:
			loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		
		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		
		if batch_idx%200==0:
			print(' Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))
		
	return train_loss/(batch_idx+1), 100.*correct.item()/total

def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	
	ece_criterion = _ECELoss().cuda()
	logits_list = []
	labels_list = []

	with torch.no_grad():
		for batch_idx, (inputs, targets,_) in enumerate(testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
		
			test_loss += loss.data.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
		
			logits_list.append(outputs)
			labels_list.append(targets)


	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss/len(testloader), correct, total,100. * correct.item() / total))

	logits = torch.cat(logits_list).cuda()
	labels = torch.cat(labels_list).cuda()
	ece_loss = ece_criterion(logits, labels).detach().item()

	print (f"ece loss {ece_loss}")

	return test_loss/len(testloader), 100. * correct.item() / total, ece_loss

loss_list = []
acc_list = []
for epoch in range(args.epochs):
    train_loss,train_acc = train(epoch)
    test_loss,test_acc,ece_loss = test(epoch)
    if (epoch%50)+1>47: # save 3 models per cycle
        print('save!')
        net.cpu()
        torch.save(net.state_dict(),args.dir + f'/{args.dataset}_{model_name}_{args.batch_size}_{datasize}%i.pt'%(mt))
        mt +=1
        net.cuda(device_id)
    
    loss_list.append((train_loss,test_loss,ece_loss))
    acc_list.append((train_acc,test_acc))
    
np.save(f'csgmcmc_{args.dataset}_{model_name}_{lr_0}_{args.batch_size}_acc.npy',np.array(acc_list))
np.save(f'csgmcmc_{args.dataset}_{model_name}_{lr_0}_{args.batch_size}_loss.npy',np.array(loss_list))