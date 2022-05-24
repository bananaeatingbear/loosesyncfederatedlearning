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
from model_utils import *
from model_utils import _batchnorm_to_groupnorm_new
from diffmi_attack import diffmi_attack
from nasr_fed_attack import nasr_fed_attack
from multi_party_attack import *


def get_naming_mid_str():
	name_string_mid_str = str(args.user_number) + '_' + str(args.num_step) + '_' + str(args.target_epochs) + '_'
	
	if (args.local_epochs > 1):
		name_string_mid_str = name_string_mid_str + str(args.local_epochs) + '_'
	
	name_string_mid_str = name_string_mid_str  + 'particle_'
	
	return name_string_mid_str


def assign_part_dataset(dataset, user_list=[]):
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
			'fashion_product' in dataset.dataset_name) or dataset.dataset_name == 'intel'):
		transform_train = transforms.ToTensor()
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
		print("1D dataset!")
	
	num_users = len(user_list)
	
	### generating train / test indices for each user
	
	## for CIFAR-10 / CIFAR-100, we have 50000/10000 train/test data,
	## then each user should share the test data and we need to split the training data
	
	## for purchase and texas, we have enough data
	
	## the maximum number of user we can support is 10
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	
	assigned_index = []
	
	'''
	attacker_evaluation_data = []
	attacker_evaluation_label = []
	attacker_evaluation_data_index = []
	#### do we need to include non-member data in this? do we need to do gradient ascent for non-members?

	### active attackers may attack in the following 2 different ways.
	### 2 cases:
	# 1: do gradient ascent for members and non-members. the grad of members should be smaller than non-members.
	# but the grad of members might be close to the grad of normal testing samples.
	# the grad of non-members would be super large.

	# 2: do gradient ascent for members only. here the goal is to tell the members apart from normal testing samples.
	# the expected result is that the grad norm of members should be super large, even larger than normal testing samples.
	# but this is actually depending on the testing accuracy. and the grad norm of members might not be that large.

	In our current experiment, we choose method 1, i.e. do gradient ascent for both members and non-members.
	We need to explicitly find a targeted set to do gradient ascent. The server is the attacker to do gradient ascent.
	'''
	
	for i in range(num_users):
		this_user = user_list[i]
		this_user.target_transform = target_transform
		this_user.train_transform = transform_train
		this_user.test_transform = transform_test
		
		this_user_index = np.random.choice(len(index_left), training_set_size, replace=False)
		this_user_train_index = index_left[this_user_index]
		new_index_left = np.setdiff1d(np.arange(len(index_left)), this_user_index)
		index_left = index_left[new_index_left]
		
		this_user.train_data = dataset.train_data[this_user_train_index]
		this_user.train_label = dataset.train_label[this_user_train_index]

		print (np.bincount(this_user.train_label))
		this_user.class_weight = np.ones((len(np.unique(this_user.train_label)))) * training_set_size / (len(np.unique(this_user.train_label)) * np.bincount(this_user.train_label))
		print ("class weight:",this_user.class_weight)
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
		### take a fraction of training data to be used as MI evaluation data (this is the member part of evaluation data)
		# when active attacker is not activated, here the eval_data_size == target_train_size
		# when active attacker is activated, here the eval_data_size == 100 or 50
		### if we are going to do targeted class GA mislabeling, then we need to make sure eval set in a specific class
		
		eval_data_index = np.random.choice(len(this_user_train_index), args.eval_data_size, replace=False)
		evaluation_data = copy.deepcopy(this_user.train_data[eval_data_index])
		evaluation_label = copy.deepcopy(this_user.train_label[eval_data_index])
		
		#### create dataset and dataloader
		train = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=True, transform=transform_train,
		                             target_transform=target_transform)
		test = part_pytorch_dataset(this_user.test_data, this_user.test_label, train=False, transform=transform_test,
		                            target_transform=target_transform)
		### training data processing includes adding noise, but when evaluating, we should remove the effect of noise
		### the data in train_eval set is the same as the train set, but the preprocessing part is removed
		train_eval = part_pytorch_dataset(this_user.train_data, this_user.train_label, train=False,
		                                  transform=transform_test, target_transform=target_transform)
		
		this_user.train_dataset = train
		this_user.test_dataset = test
		this_user.train_eval_dataset = train_eval
		
		this_user.train_data_loader = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                          shuffle=True, num_workers=1)
		this_user.test_data_loader = torch.utils.data.DataLoader(test, batch_size=args.target_batch_size, shuffle=False,
		                                                         num_workers=1)
		this_user.train_eval_data_loader = torch.utils.data.DataLoader(train_eval, batch_size=1, shuffle=False,
		                                                               num_workers=1)
		this_user.train_loader_in_order = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                              shuffle=False, num_workers=1)
		
		evaluation = part_pytorch_dataset(evaluation_data, evaluation_label, train=False, transform=transform_test,
		                                  target_transform=target_transform)
		this_user.evaluation_member_dataset = evaluation
		
		### we use testing data as 'eval_non_member' set
		non_member_index = np.random.choice(len(this_user.test_label), args.eval_data_size, replace=False)
		evaluation_non_member = part_pytorch_dataset(copy.deepcopy(this_user.test_data[non_member_index]),
		                                             copy.deepcopy(this_user.test_label[non_member_index]), train=False,
		                                             transform=transform_test,
		                                             target_transform=target_transform)
		this_user.evaluation_non_member_dataset = evaluation_non_member
	
	### check remaining unassigned data
	dataset.remaining_index = index_left
	print(len(index_left))
	
	### we select some data as validation set
	validation_data_index = np.random.choice(index_left, len(index_left),
	                                         replace=False)  ### this should be false, but just for the sake of # of user exp
	validation_data = dataset.train_data[validation_data_index]
	validation_label = dataset.train_label[validation_data_index]
	dataset.remaining_index = np.setdiff1d(index_left, validation_data_index)
	
	for user_idx in range(num_users):
		this_user = user_list[user_idx]
		this_user.eval_validation_data = validation_data
		this_user.eval_validation_label = validation_label
		this_user.validation_dataset = part_pytorch_dataset(validation_data, validation_label, train=False,
		                                                    transform=transform_test,
		                                                    target_transform=target_transform)
		this_user.validation_data_loader = torch.utils.data.DataLoader(this_user.validation_dataset,
		                                                               batch_size=args.target_batch_size, shuffle=False,
		                                                               num_workers=1)
		### processing validation set for MMD defense
		
		### sort the validation data according to the class index
		sorted_index = np.argsort(this_user.eval_validation_label)
		this_user.eval_validation_data = this_user.eval_validation_data[sorted_index]
		this_user.eval_validation_label = this_user.eval_validation_label[sorted_index]
		
		### create a index list for starting index of each class
		this_user.starting_index = []
		# print ("starting index",self.starting_index)
		for i in np.unique(this_user.eval_validation_label):
			for j in range(len(this_user.eval_validation_label)):
				if (this_user.eval_validation_label[j] == i):
					# print ("class %d index %d "%(i,j))
					this_user.starting_index.append(j)
					break
	
	print(len(index_left))


def run_blackbox_attacks(user_list, target_model, num_classes, output_file):
	# num_classes = 10 if (args.dataset == 'cifar10' or args.dataset == 'mnist') else 100
	acc = 0
	for user_idx in range(len(user_list)):
		black_ref = blackbox_attack(args.eval_data_size, 'global_prob', num_classes=num_classes)
		total_confidences, total_classes, total_labels = black_ref.get_attack_input(target_model, user_list[user_idx])
		acc += black_ref.attack(total_confidences=total_confidences, total_classes=total_classes,
		                        total_labels=total_labels, output_file=output_file)  ### labels here is the true label
	# print ("blackbox attack finished")
	return acc / len(user_list)


def simplex_test(user_list, local_weight, local_model, epoch, num_step, comm_round_idx, test_round=10,ori_model_weight={}):
	naming_str = get_naming_mid_str() + str(epoch + 1) + '_' + str(num_step) + '_' + str(comm_round_idx) + '_' + str(
		args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
		args.model_name) + '.npy'
	
	### simplex test
	acc = []
	loss = []
	for _ in range(test_round):
		weight_param = simplex_uniform_sampling(len(user_list))
		# print (weight_param)
		weighted_avg_weight = average_weights(local_weight, weight_param)
		local_model.load_state_dict(weighted_avg_weight)
		train_acc, test_acc, train_loss, test_loss = get_train_test_acc(user_list, local_model, print_option=False,
		                                                                return_loss=True)
		bb_acc = run_blackbox_attacks(user_list, local_model, num_classes=len(np.unique(user_list[0].train_label)),
		                              output_file=None)
		acc.append((train_acc, test_acc, bb_acc))
		loss.append((train_loss, test_loss))
	
	data_name = './simplex/simplex_test_acc_' + naming_str
	acc = np.array(acc)
	np.save(data_name, acc)
	
	data_name = './simplex/simplex_test_loss_' + naming_str
	loss = np.array(loss)
	np.save(data_name, loss)
	# print (acc.shape,loss.shape,naming_str)
	
	### flatness test
	alpha_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
	acc = []
	loss = []
	
	for user_idx in range(len(user_list)):
		this_acc = []
		this_loss = []
		for this_alpha in alpha_list:
			weight_param = np.array([this_alpha, 1 - this_alpha])
			weighted_avg_weight = average_weights([ori_model_weight, local_weight[user_idx]], weight_param)
			local_model.load_state_dict(weighted_avg_weight)
			train_acc, test_acc, train_loss, test_loss = get_train_test_acc(user_list, local_model, print_option=False,
			                                                                return_loss=True)
			bb_acc = run_blackbox_attacks(user_list, local_model, num_classes=len(np.unique(user_list[0].train_label)),
			                              output_file=None)
			this_acc.append((train_acc, test_acc, bb_acc))
			this_loss.append((train_loss, test_loss))
		acc.append(this_acc)
		loss.append(this_loss)
	
	data_name = './simplex/flatness_test_acc_' + naming_str
	acc = np.array(acc)
	np.save(data_name, acc)
	
	data_name = './simplex/flatness_test_loss_' + naming_str
	loss = np.array(loss)
	np.save(data_name, loss)
	print(acc.shape, loss.shape, naming_str)


def likelihood_sampling(likelihood_list):
	### sampling from the particle list
	choice = np.random.choice(len(likelihood_list),size=1,p=np.array(likelihood_list))[0]
	#print(f"likelihood sampling choice {choice}")
	return choice

def calculate_likelihood(user_list, model, weight_list):

	from scipy.special import softmax
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
	likelihood_list = []
	for idx in range(len(user_list)):
		model.to(device)
		model.load_state_dict(weight_list[idx])
		model.eval()
		val_loss = 0
		total = 0
		for images, labels, _ in user_list[0].validation_data_loader:
			model.zero_grad()
			images = images.to(device)
			outputs = model(images)
			labels = labels.to(device)
			total += labels.size(0)
			loss = criterion(outputs, labels)
			
			#print (outputs[0],labels[0])
			#print (torch.nn.functional.softmax(outputs[0],dim=0))
			#print (softmax(outputs[0].detach().cpu().numpy()))
			#print (loss[0])
			val_loss += loss.detach().item()
			
		likelihood_list.append(val_loss)
		print (f" this user {idx}: val loss {val_loss}")
		
	print (f"total number {total}, likelihood list {likelihood_list}")
		### pytorch use natural log
	likelihood_list = np.array(likelihood_list)
	likelihood_list = softmax(-1*likelihood_list)
	#likelihood_list = likelihood_list - np.amin(likelihood_list)
	#likelihood_list = np.exp(-1*likelihood_list)
	#likelihood_list = likelihood_list/np.sum(likelihood_list)
	#print (f"likelihood list {likelihood_list}")
	return likelihood_list

def train_models(user_list, target_model, learning_rate, decay, epochs):
	num_users = len(user_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.type(torch.FloatTensor)
	target_model.to(device)
	momentum = 0.9
	
	sampling_stat = []
	volume_results = []
	loss_list = []
	acc_list = []
	likelihood_list = []

	model_list = [target_model.state_dict() for _ in range(len(user_list))]
	start_likelihood_list = [0.1 for _ in range(len(user_list))]
	likelihood_list.append(start_likelihood_list)
	
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		this_optim = torch.optim.SGD(user_list[user_idx].model.parameters(), lr=learning_rate, momentum=momentum,
		                             weight_decay=decay)
		user_list[user_idx].optim = this_optim
	
	### start training
	for epoch in tqdm(range(int(epochs / args.local_epochs))):
		
		## LR schedule
		if (epoch in args.schedule):
			learning_rate = learning_rate / 10
			print("new learning rate = %f" % (learning_rate))
			### set up new learning rate for each user
			for user_idx in range(num_users):
				for param_group in user_list[user_idx].optim.param_groups:
					param_group['lr'] = learning_rate
		
		#### we need to know the # of steps, # of training set size to perform fed sgd
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		
		for comm_round_idx in range(comm_round_per_epoch):
			
			local_weights = []
			attack_loader_list = []
			this_round_sampling = []
			### get parameter update from each client
			
			for idx in range(len(user_list)):
				
				# create a new dataloader
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				
				# randomly sample based on likelihood
				sampled_idx = likelihood_sampling(start_likelihood_list)
				this_round_sampling.append(sampled_idx)
				
				w, ban_list = update_weights(current_model_weights=model_list[sampled_idx],
				                             model=user_list[idx].model, optimizer=user_list[idx].optim,
				                             train_loader=train_data_loader, local_epochs=args.local_epochs,
				                             mixup=args.mixup, num_step=args.num_step,class_weights=user_list[idx].class_weight)
				attack_loader_list.append(user_list[idx].create_batch_attack_data_loader(ban_list))
				user_list[idx].update_ban_list(ban_list)
				local_weights.append(copy.deepcopy(w))
			
			### calculate volume
			volume_weights = copy.deepcopy(local_weights)
			prev_volume_weights = [model_list[x] for x in this_round_sampling]
			volume_results.append(calculate_volume(volume_weights, prev_volume_weights, lr=learning_rate, num_step=args.num_step))
			
			### record stats
			print (f"likelihood this round {start_likelihood_list}, sampling stat this round {this_round_sampling}")
			start_likelihood_list = calculate_likelihood(user_list,copy.deepcopy(target_model),local_weights)
			sampling_stat.append(this_round_sampling)
			likelihood_list.append(start_likelihood_list)
			
			### record loss
			for idx,this_model_weights in enumerate(local_weights):
				local_model = copy.deepcopy(target_model)
				local_model.load_state_dict(this_model_weights)
				train_acc, test_acc, train_loss, test_loss = get_train_test_acc(user_list, local_model,return_loss=True,print_option=False)
				loss_list.append((train_loss,test_loss))
				acc_list.append((train_acc,test_acc))
				
				print (f" user{idx}, acc:{train_acc}, {test_acc}, loss:{train_loss}, {test_loss}")
			
			### update model
			model_list = copy.deepcopy(local_weights)
	
	volume_name_str = './simplex/volume_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	volume_results = np.array(volume_results)
	np.save(volume_name_str, volume_results)
	
	loss_name_str = './simplex/loss_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	np.save(loss_name_str, np.array(loss_list))
	
	acc_name_str = './simplex/acc_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	np.save(acc_name_str, np.array(acc_list))
	
	likelihood_name_str = './simplex/likelihood_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	np.save(likelihood_name_str, np.array(likelihood_list))
	
	sampling_stat_name_str = './simplex/sampling_stat_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	np.save(sampling_stat_name_str, np.array(sampling_stat))
	

	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"train acc {train_acc},test acc {test_acc}")
	
	return target_model, train_acc, test_acc


def attack_experiment():
	import warnings
	warnings.filterwarnings("ignore")
	np.random.seed(seed=12345)
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	### dataset && membership inference data
	membership_attack_number = args.membership_attack_number
	target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
	                         membership_attack_number=membership_attack_number,
	                         cutout=args.cutout, n_holes=args.n_holes, length=args.length)
	
	print(np.bincount(target_dataset.label))
	
	user_list = [User(dataset=args.dataset, model_name=args.target_model, id=i) for i in range(args.user_number)]
	
	assign_part_dataset(target_dataset, user_list)
	
	if (target_dataset.dataset_name == 'cifar100'):
		args.num_classes = 100
		if (args.model_name == 'resnet20'):
			### here, if we replace the layers, the train / test acc curve is different from the original.
			### the training will slow down
			target_model = resnet(depth=20, num_classes=100)
		# target_model = convert_batchnorm_modules(resnet(depth=20, num_classes=100),converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'densenet_cifar'):
			# print (densenet(depth=100,num_classes=100))
			target_model = convert_batchnorm_modules(densenet(depth=100, num_classes=100),
			                                         converter=_batchnorm_to_groupnorm_new)
		# print (target_model)
		if (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=100)
	
	elif (target_dataset.dataset_name == 'cifar10'):
		args.num_classes = 10
		if (args.model_name == 'resnet20'):
			target_model = resnet(depth=20, num_classes=10)
		# target_model = convert_batchnorm_modules(resnet(depth=20, num_classes=100), converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'densenet_cifar'):
			target_model = convert_batchnorm_modules(densenet(depth=100, num_classes=10),
			                                         converter=_batchnorm_to_groupnorm_new)
		if (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=10)
	elif ('celeba' in target_dataset.dataset_name):
		args.num_classes = 2
		target_model = celeba_model()
	else:
		target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
		if (target_dataset.dataset_name == 'texas'):
			args.num_classes = 100
		if (target_dataset.dataset_name == 'purchase'):
			args.num_classes = 100
	
	# print (target_model)
	#pytorch_total_params = sum(p.numel() for p in target_model.parameters())
	#print(f"total params {pytorch_total_params}")
	
	target_model, train_acc, test_acc = train_models(user_list, target_model, learning_rate=args.target_learning_rate,
	                                                 decay=args.target_l2_ratio, epochs=args.target_epochs)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_data_size', type=int, default=3000)
	parser.add_argument('--target_model', type=str, default='cnn')
	parser.add_argument('--target_learning_rate', type=float, default=0.01)
	parser.add_argument('--attack_learning_rate', type=float, default=0.001)
	parser.add_argument('--target_batch_size', type=int, default=100)
	parser.add_argument('--attack_batch_size', type=int, default=100)
	parser.add_argument('--target_epochs', type=int, default=20)
	parser.add_argument('--attack_epochs', type=int, default=500)
	parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
	parser.add_argument('--shadow_data_size', type=int, default=30000)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--num_classes', type=int, default=10)
	# parser.add_argument('--attack_times', type=int, default=1)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--membership_attack_number', type=int, default=0)
	parser.add_argument('--reference_number', type=int, default=1)
	
	parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120])
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--alpha', type=float, default='1.0')
	parser.add_argument('--mixup', type=int, default=0)
	# parser.add_argument('--dropout', type=int, default=0)
	parser.add_argument('--cutout', type=int, default=0)
	parser.add_argument('--n_holes', type=int, default=1)
	parser.add_argument('--length', type=int, default=16)
	
	parser.add_argument('--num_step', type=int, default=20)
	### fed params
	parser.add_argument('--local_epochs', type=int, default=1)
	parser.add_argument('--user_number', type=int, default=2)
	parser.add_argument('--noniid', type=int, default=0)
	parser.add_argument('--eval_data_size', type=int, default=50)
	
	parser.add_argument('--random_seed', type=int, default=123)
	
	args = parser.parse_args()
	print(vars(args))
	
	import torch
	torch.manual_seed(args.random_seed)
	import numpy as np
	np.random.seed(args.random_seed)
	import sklearn
	sklearn.utils.check_random_state(args.random_seed)
	attack_experiment()
	
	print(vars(args))
