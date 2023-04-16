
import warnings
from collections import OrderedDict

import random
import sklearn
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from model import *
from data import *
from user import *
#warnings.filterwarnings("ignore", category=UserWarning)



def get_model_params(model: "PyTorch Model"):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
	def __init__(self):
		self.train_data = None
		self.train_label = None
		self.test_data = None
		self.test_label = None
		self.valid_data = None
		self.valid_label = None
		self.available_list = None

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if (args.dataset == 'cifar10'):
			#self.model = ResNet18(num_classes=10)
			self.model = alexnet(num_classes=10)
		else:
			self.model = TargetNet(args.dataset, 0, 0)
		self.model.to(device)
			
		self.optimizer =  torch.optim.SGD(self.model.parameters(), lr=args.target_learning_rate, momentum=0.9,weight_decay=args.target_l2_ratio)
		
		### read data from assigned partition. all clients share the same test / valid set
		self.train_data = np.load(
			f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_{args.user_id}_train_data.npy')
		self.train_label = np.load(
			f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_{args.user_id}_train_label.npy')
		self.test_data = np.load(
			f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_test_data.npy')
		self.test_label = np.load(
			f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_test_label.npy')
		self.valid_data = np.load(
			f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_valid_data.npy')
		self.valid_label = np.load(
			f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_valid_label.npy')

		#print(np.bincount(this_user.train_label))
		self.class_weights = np.ones((len(np.unique(self.train_label)))) * args.target_data_size / (len(np.unique(self.train_label)) * np.bincount(self.train_label))
		#print("class weight:", this_user.class_weight)
		
		self.available_list = np.arange(len(self.train_label))
		
		if (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
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
		
		if (args.dataset == 'mnist' or args.dataset == 'fashion_mnist' or (
				'celeb' in args.dataset) or args.dataset == 'sat6' or args.dataset == 'retina' or (
				'fashion_product' in args.dataset) or args.dataset == 'intel' or args.dataset == 'gtsrb'):
			transform_train = transforms.ToTensor()
			transform_test = transforms.ToTensor()
			target_transform = transforms.ToTensor()
			
		self.train_transform = transform_train
		self.test_transform = transform_test
		self.target_transform = target_transform
			
		train = part_pytorch_dataset(self.train_data, self.train_label, train=True, transform=transform_train,target_transform=target_transform)
		test = part_pytorch_dataset(self.test_data, self.test_label, train=False, transform=transform_test,target_transform=target_transform)
		valid = part_pytorch_dataset(self.valid_data, self.valid_label, train=False, transform=transform_test,target_transform=target_transform)
		
		self.train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,shuffle=True, num_workers=1)
		self.test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,num_workers=1)
		self.valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=args.target_batch_size, shuffle=False,num_workers=1)
	
	def create_new_train_data_loader(self, batch_size):
		if (self.available_list is None):
			self.available_list = np.arange(len(self.train_data))
		new_train_dataset = part_pytorch_dataset(self.train_data[self.available_list],self.train_label[self.available_list], train=True,transform=self.train_transform,target_transform=self.target_transform)
		new_train_data_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size,shuffle=True, num_workers=1)
		return new_train_data_loader
	
	def update_ban_list(self, ban_list):
		self.available_list = np.setdiff1d(self.available_list, self.available_list[ban_list])
		if (len(self.available_list) == 0):
			self.reset_ban_list()
	
	def reset_ban_list(self):
		self.available_list = np.arange(len(self.train_data))
	
	def get_parameters(self):
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
	
	def set_parameters(self, parameters):
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
		self.model.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config=None):
		self.set_parameters(parameters)
		used_index = self.update_weights(train_loader=self.create_new_train_data_loader(batch_size=args.target_batch_size),local_epochs=args.local_epochs,num_step=args.num_step,class_weights=self.class_weights)
		self.update_ban_list(used_index)
		#print ("fit once")
		return self.get_parameters(), args.num_step, {}
	
	def evaluate(self, parameters, config):
		self.set_parameters(parameters)
		#valid_loss,valid_acc = self.test_model(self.valid_data_loader)
		test_loss,test_acc = self.test_model(self.test_data_loader)
		return test_loss, int(len(self.test_label)/args.target_batch_size), {"accuracy": test_acc}
	
	def test_model(self, data_loader):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		criterion = torch.nn.CrossEntropyLoss()
		correct, total, loss = 0, 0, 0.0
		self.model.to(device).eval()
		with torch.no_grad():
			for images, labels,_ in tqdm(data_loader):
				outputs = self.model(images.to(device))
				labels = labels.to(device)
				loss += criterion(outputs, labels).item()
				total += labels.size(0)
				correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
		return loss / len(data_loader.dataset), correct / total

	def update_weights(self, train_loader, local_epochs, num_step=10,class_weights=None, unequal=0):
		# Set mode to train model
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		class_weights = torch.from_numpy(class_weights).float()
		criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
		self.model.train()
		self.model.to(device)
		self.model.zero_grad()
		### for each user, we need to create a new dataloader, so we can avoid using instances that are used in previous steps, same epoch.
		### for single worker, it is not possible to play with sampler to achieve the above constraint.
		all_data_idx = []
		step_count = 0
		
		for iter in range(local_epochs):
			for batch_idx, (images, labels, data_idx) in enumerate(train_loader):
				images, labels = images.to(device), labels.to(device)
				self.model.zero_grad()
				log_probs = self.model(images)
				loss = criterion(log_probs, labels)
				loss.backward()
				self.optimizer.step()
			
				data_idx, _ = torch.sort(data_idx)
				all_data_idx.append(data_idx)
			
				step_count += 1
				if (step_count == num_step and local_epochs == 1 and (not unequal)):
					# print (f"time to break")
					break
	
		all_data_idx = torch.unique(torch.hstack(all_data_idx))

		return all_data_idx

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_learning_rate', type=float, default=0.01)
	parser.add_argument('--target_data_size', type=int, default=0.01)
	parser.add_argument('--target_batch_size', type=int, default=100)
	parser.add_argument('--target_epochs', type=int, default=20)
	parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--model_name', type=str, default='resnet18')
	parser.add_argument('--num_step', type=int, default=20)
	parser.add_argument('--local_epochs', type=int, default=1)
	parser.add_argument('--random_seed', type=int, default=3)
	parser.add_argument('--user_id', type=int, default=0)
	args = parser.parse_args()
	print(vars(args))
	
	###  set up random seed
	torch.manual_seed(args.random_seed)
	cudnn.benchmark = True
	cudnn.deterministic = True
	np.random.seed(args.random_seed)
	sklearn.utils.check_random_state(args.random_seed)
	random.seed(args.random_seed)
	
	client = FlowerClient()
	fl.client.start_numpy_client("0.0.0.0:8080", client=client)