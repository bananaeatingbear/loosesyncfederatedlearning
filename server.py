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
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics

def get_model_params(model: "PyTorch Model"):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def assign_part_dataset(dataset,user_num=10):
	
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	assigned_index = []
	
	for i in range(user_num):
		this_user_index = np.random.choice(len(index_left), training_set_size, replace=False)
		this_user_train_index = index_left[this_user_index]
		new_index_left = np.setdiff1d(np.arange(len(index_left)), this_user_index)
		index_left = index_left[new_index_left]
		this_user_train_data = dataset.train_data[this_user_train_index]
		this_user_train_label = dataset.train_label[this_user_train_index]
		assigned_index.append(this_user_train_index)
		
		np.save(f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_{i}_train_data.npy',this_user_train_data)
		np.save(f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_{i}_train_label.npy',this_user_train_label)
	
	### check remaining unassigned data
	dataset.remaining_index = index_left
	valid_data = dataset.train_data[index_left]
	valid_label = dataset.train_label[index_left]
	
	np.save(f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_test_data.npy',dataset.test_data)
	np.save(f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_test_label.npy',dataset.test_label)
	np.save(f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_valid_data.npy',valid_data)
	np.save(f'./fed_data/{args.dataset}_{args.model_name}_{args.target_data_size}_{args.num_step}_valid_label.npy',valid_label)
	
	### prepare data for csgmcmc
	assigned_index = np.array(assigned_index).flatten()
	np.save(f'./csgmcmc/{dataset.dataset_name}_{args.target_data_size}_train_data.npy', dataset.train_data[assigned_index])
	np.save(f'./csgmcmc/{dataset.dataset_name}_{args.target_data_size}_train_label.npy',dataset.train_label[assigned_index])
	np.save(f'./csgmcmc/{dataset.dataset_name}_{args.target_data_size}_test_data.npy', dataset.test_data)
	np.save(f'./csgmcmc/{dataset.dataset_name}_{args.target_data_size}_test_label.npy', dataset.test_label)
	
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
	
	test = part_pytorch_dataset(dataset.test_data, dataset.test_label, train=False, transform=transform_test,target_transform=target_transform)
	valid = part_pytorch_dataset(valid_data,valid_label, train=False, transform=transform_test,target_transform=target_transform)
	
	test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,num_workers=1)
	valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=args.target_batch_size, shuffle=False,num_workers=1)
	
	return test_data_loader,valid_data_loader


def test_model(model,data_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = torch.nn.CrossEntropyLoss()
	correct, total, loss = 0, 0, 0.0
	model.eval()
	with torch.no_grad():
		for images, labels,_ in tqdm(data_loader):
			outputs = model(images.to(device))
			labels = labels.to(device)
			loss += criterion(outputs, labels).item()
			total += labels.size(0)
			correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
	return loss / len(data_loader.dataset), correct / total

#def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    #accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    #examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    #return {"accuracy": sum(accuracies) / sum(examples)}
#    return {"accuracy": 1}

def get_eval_result(weights):
	global min_valid_loss
	state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
	model.load_state_dict(state_dict, strict=True)
	model.eval()
	valid_loss,valid_acc = test_model(model,valid_data_loader)
	if (valid_loss < min_valid_loss):
		min_valid_loss = valid_loss
		model_path = f"./model_checkpoints/{args.model_name}_{args.dataset}_{args.num_step}_{args.user_number}_{args.target_data_size}_{0}_{args.local_epochs}.pt"
		torch.save(model.state_dict(), model_path)
	test_loss,test_acc = test_model(model,test_data_loader)
	return [test_loss,{"test_accuracy":test_acc,"valid_loss":valid_loss,"valid_acc":valid_acc}]
	

parser = argparse.ArgumentParser()
parser.add_argument('--target_data_size', type=int, default=3000)
parser.add_argument('--target_model', type=str, default='cnn')
parser.add_argument('--target_learning_rate', type=float, default=0.01)
parser.add_argument('--target_batch_size', type=int, default=100)
parser.add_argument('--target_epochs', type=int, default=20)
parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120])
parser.add_argument('--model_name', type=str, default='alexnet')
parser.add_argument('--num_step', type=int, default=20)
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--user_number', type=int, default=2)
parser.add_argument('--random_seed', type=int, default=3)
	
args = parser.parse_args()
print(vars(args))
###  set up random seed
torch.manual_seed(args.random_seed)
cudnn.benchmark = True
cudnn.deterministic = True
np.random.seed(args.random_seed)
sklearn.utils.check_random_state(args.random_seed)
random.seed(args.random_seed)

### set up model
if (args.dataset == 'cifar10'):
	#model = ResNet18(num_classes=10)
	model = alexnet(num_classes=10)
else:
	model = TargetNet(args.dataset, 0, 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
### set up dataset
target_dataset = dataset(dataset_name=args.dataset, gpu=1,membership_attack_number=0)
	
test_data_loader,valid_data_loader = assign_part_dataset(target_dataset,args.user_number)

print ("data assignment finished")
global min_valid_loss
min_valid_loss = 1e20

# Define strategy
strategy = fl.server.strategy.FedAvg(fraction_fit=1,min_fit_clients=args.user_number,min_available_clients=args.user_number,initial_parameters=fl.common.weights_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()]),eval_fn=get_eval_result)
## evaluate_metrics_aggregation_fn=weighted_average
	
# Start Flower server
fl.server.start_server(server_address="0.0.0.0:8080",config={"num_rounds": args.target_epochs},strategy=strategy)