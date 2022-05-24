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

from model import *
from torch.autograd import Variable
import numpy as np
import random
from data import *
from model_utils import _ECELoss
from scipy.stats import entropy

ece_criterion = _ECELoss().cuda()

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Ensemble')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--device_id', type=int, help='device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

net = TargetNet('fashion_mnist', 1, 10)

net.cuda(device_id)
if use_cuda:
	net.cuda(device_id)
	cudnn.benchmark = True
	cudnn.deterministic = True
criterion = nn.CrossEntropyLoss()

fashion_dataset = dataset(dataset_name='fashion_mnist', gpu=1, membership_attack_number=0)
transform_test = transforms.Compose([
	transforms.ToTensor(),
])

cifar_testset = part_pytorch_dataset(fashion_dataset.test_data, fashion_dataset.test_label, train=False,
                                     transform=transform_test, target_transform=None)
cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=100, shuffle=False, num_workers=0)

train_data = np.load('./csgmcmc/fashion_mnist_2000_train_data.npy')
train_label = np.load('./csgmcmc/fashion_mnist_2000_train_label.npy')
train_set = part_pytorch_dataset(train_data, train_label, train=False, transform=transform_test, target_transform=None)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=False, num_workers=0)

target_model_class_num = len(np.unique(cifar_testset.test_labels))

target_dataset = dataset(dataset_name='mnist', gpu=1, membership_attack_number=0)
transform_test = transforms.ToTensor()
target_transform = transforms.ToTensor()
sampled_class = np.sort(np.random.choice(np.unique(target_dataset.test_label), target_model_class_num, replace=False))
sampled_test_index = []
sampled_test_label = []
for idx, this_class in enumerate(sampled_class):
	this_class_index = np.arange(len(target_dataset.test_label))[target_dataset.test_label == this_class]
	sampled_test_index.append(this_class_index)
	sampled_test_label.append(np.ones(len(this_class_index)) * idx)
sampled_test_index = np.hstack(sampled_test_index)
sampled_test_data = target_dataset.test_data[sampled_test_index, :, :, :]
sampled_test_label = np.hstack(sampled_test_label)

gtsrb_testset = part_pytorch_dataset(sampled_test_data, sampled_test_label, train=False,
                                     transform=transform_test,
                                     target_transform=target_transform)
gtsrb_testloader = torch.utils.data.DataLoader(gtsrb_testset, batch_size=100, shuffle=False, num_workers=0)


def get_accuracy(truth, pred):
	assert len(truth) == len(pred)
	right = 0
	for i in range(len(truth)):
		if truth[i] == pred[i]:
			right += 1.0
	return right / len(truth)


def ece_score(py, y_test, n_bins=10):
	py = np.array(py)
	y_test = np.array(y_test)
	if y_test.ndim > 1:
		y_test = np.argmax(y_test, axis=1)
	py_index = np.argmax(py, axis=1)
	py_value = []
	for i in range(py.shape[0]):
		py_value.append(py[i, py_index[i]])
	py_value = np.array(py_value)
	acc, conf = np.zeros(n_bins), np.zeros(n_bins)
	Bm = np.zeros(n_bins)
	for m in range(n_bins):
		a, b = m / n_bins, (m + 1) / n_bins
		for i in range(py.shape[0]):
			if py_value[i] > a and py_value[i] <= b:
				Bm[m] += 1
				if py_index[i] == y_test[i]:
					acc[m] += 1
				conf[m] += py_value[i]
		if Bm[m] != 0:
			acc[m] = acc[m] / Bm[m]
			conf[m] = conf[m] / Bm[m]
	ece = 0
	for m in range(n_bins):
		ece += Bm[m] * np.abs((acc[m] - conf[m]))
	return ece / sum(Bm)


def train():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	pred_list = []
	true_label = []
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(trainloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			true_label.append(targets)
			outputs = net(inputs)
			pred_list.append(F.softmax(outputs, dim=1))
			loss = criterion(outputs, targets)
			
			test_loss += loss.data.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
	
	# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
	#    test_loss/len(cifar_testloader), correct, total,
	#    100. * correct.item() / total))
	pred_list = torch.cat(pred_list, 0)
	true_label = torch.stack(true_label).flatten()
	return pred_list, true_label


def test():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	pred_list = []
	true_label = []
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(cifar_testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			true_label.append(targets)
			outputs = net(inputs)
			pred_list.append(F.softmax(outputs, dim=1))
			loss = criterion(outputs, targets)
			
			test_loss += loss.data.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
	
	# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
	#    test_loss/len(cifar_testloader), correct, total,
	#    100. * correct.item() / total))
	pred_list = torch.cat(pred_list, 0)
	true_label = torch.cat(true_label).flatten()
	return pred_list, true_label


def test_uncertainty():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	pred_list = []
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(gtsrb_testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
			outputs = net(inputs)
			pred_list.append(F.softmax(outputs, dim=1))
			loss = criterion(outputs, targets)
			
			test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
	
	# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
	#    test_loss/len(testloader), correct, total,
	#    100. * correct.item() / total))
	pred_list = torch.cat(pred_list, 0)
	return pred_list


def test_ensemble_model(model_name_list, exp_name='CSGMCMC'):
	print("----------")
	
	pred_list = []
	num_model = len(model_name_list)
	for m in range(num_model):
		# print (m,exp_name)
		if (m == 0 and exp_name != 'CSGMCMC'):
			checkpoint = torch.load(model_name_list[m])
			net.load_state_dict(checkpoint['model_state_dict'])
		else:
			net.load_state_dict(torch.load(model_name_list[m]))
		pred, truth_res = train()
		pred_list.append(pred)
	
	fake = sum(pred_list) / num_model
	values, pred_label = torch.max(fake, dim=1)
	pred_res = list(pred_label.data)
	acc = get_accuracy(truth_res, pred_res)
	print(f"{exp_name} train acc", acc)
	
	sum_loss = 0
	for i in range(len(fake)):
		this_pred = fake[i]
		this_label = truth_res[i]
		this_loss = -1 * torch.log(this_pred[this_label])
		sum_loss += this_loss.item()
	print(f"{exp_name} avg train loss", sum_loss / len(fake))
	
	pred_list = []
	num_model = len(model_name_list)
	for m in range(num_model):
		# print (m,exp_name)
		if (m == 0 and exp_name != 'CSGMCMC'):
			checkpoint = torch.load(model_name_list[m])
			net.load_state_dict(checkpoint['model_state_dict'])
		else:
			net.load_state_dict(torch.load(model_name_list[m]))
		pred, truth_res = test()
		pred_list.append(pred)
	
	fake = sum(pred_list) / num_model
	values, pred_label = torch.max(fake, dim=1)
	pred_res = list(pred_label.data)
	acc = get_accuracy(truth_res, pred_res)
	print(f"{exp_name} test acc", acc)
	
	sum_loss = 0
	for i in range(len(fake)):
		this_pred = fake[i]
		this_label = truth_res[i]
		this_loss = -1 * torch.log(this_pred[this_label])
		sum_loss += this_loss.item()
	print(f"{exp_name} avg test loss", sum_loss / len(fake))
	
	### ECE
	# print (fake.size(),truth_res.size())
	fake = fake.cpu().numpy()
	truth_res = truth_res.cpu().numpy()
	ece = ece_score(fake, truth_res, n_bins=10)
	print(f"{exp_name} ECE score {ece}")
	
	pred_list = []
	for m in range(num_model):
		# print (m,exp_name)
		if (m == 0 and exp_name != 'CSGMCMC'):
			checkpoint = torch.load(model_name_list[m])
			net.load_state_dict(checkpoint['model_state_dict'])
		else:
			net.load_state_dict(torch.load(model_name_list[m]))
		pred = test_uncertainty()
		pred_list.append(pred)
	
	fake = sum(pred_list) / num_model
	sum_loss = 0
	sum_loss_list = []
	for i in range(len(fake)):
		this_pred = fake[i]
		this_label = truth_res[i]
		this_loss = entropy(this_pred.cpu().numpy())
		sum_loss += this_loss
		sum_loss_list.append(this_loss)
	print(f"{exp_name} avg uncertainty", sum_loss / len(fake))
	np.save(f'./{exp_name}_uncertainty_fashion_mnist.npy', np.array(sum_loss_list))
	
	if (len(model_name_list) > 1):
		from model_utils import calculate_volume
		model_weight_list = []
		for m in range(num_model):
			# print (m,exp_name)
			if (m == 0 and exp_name != 'CSGMCMC' and exp_name):
				checkpoint = torch.load(model_name_list[m])
				model_weight_list.append(checkpoint['model_state_dict'])
			else:
				model_weight_list.append(torch.load(model_name_list[m]))
		zero_prev = {}
		for key, val in model_weight_list[0].items():
			zero_prev[key] = torch.zeros_like(val)
		
		# if (exp_name == 'CSGMCMC'):
		#	_, volume = calculate_volume(model_weight_list=model_weight_list[:-1], prev_weight_list=[zero_prev], lr=1,max_dim=9)
		# else:
		_, volume = calculate_volume(model_weight_list=model_weight_list, prev_weight_list=[zero_prev], lr=1, max_dim=8)
		print(f"volume test {volume}")
	print("----------")


model = 'fashion_mnist'
dataset = 'fashion_mnist'

### csgmcmc
num_model = 12  ## this should be 12
model_name_list = []
for m in range(num_model):
	model_name_list.append(args.dir + '/fashion_mnist_fashion_mnist_64_20000%i.pt' % (m))
test_ensemble_model(model_name_list, exp_name='CSGMCMC')

num_model = 10
for step in range(1, 6):
	model_name_list = [f'./model_checkpoints/{model}_{dataset}_5_10_2000_0.pt']
	# {args.model_name}_{args.dataset}_{args.num_step}_{args.user_number}_{args.target_data_size}_{user_idx}.pt"
	for m in range(num_model):
		model_name_list.append(f'./model_checkpoints/{model}_{dataset}_5_10_2000_{m}_mcmc_{step}.pt')
	test_ensemble_model(model_name_list, exp_name=f'FED-{step}round10ensemble')

model_name_list = [f'./model_checkpoints/{model}_{dataset}_5_10_2000_0.pt']
test_ensemble_model(model_name_list, exp_name=f'fed-avg')

model_name_list = [f'./model_checkpoints/{model}_{dataset}_200_1_20000_0.pt']
test_ensemble_model(model_name_list, exp_name=f'sgd')