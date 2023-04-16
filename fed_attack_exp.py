import model
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
from model_utils import get_train_loss
from multi_party_attack import *
import datetime
import gc
from run_attacks import *
from signum import Signum,SignedAdam
import torch.hub
from opacus.validators import ModuleValidator
from model_utils import skin_special


def get_naming_mid_str():
	name_string_mid_str =  str(args.random_seed) + '_' + str(args.noniid) + '_' + ('client_' if (args.client_adversary) else 'server_') + \
	                      (str(args.active_attacker)) + '_' + (str(args.active_attacker_lr_multiplier)) + '_' + \
	                      str(args.user_number) + '_' + str(args.num_step) + '_' + str(args.dpsgd) + '_' + str(
		args.noise_scale) + '_' + str(args.grad_norm) + '_' + str(args.mmd) + '_' + str(
		args.mmd_loss_lambda) + '_' + str(args.mixup) + '_'
	
	if(args.signsgd):
		name_string_mid_str = 'sign_' + name_string_mid_str
	
	if (args.test_rank!=0):
		name_string_mid_str = str(args.test_rank) + '_' + name_string_mid_str
	### add a string for utility preserving but actually no constraint case
	
	if (args.data_aug!=0):
		name_string_mid_str = str(args.data_aug) + '_' + name_string_mid_str
		
	if (args.whole_nn!=0):
		name_string_mid_str = 'whole_nn_' + name_string_mid_str
		
	if (args.model_name == 'resnet18' and args.num_kernels!=0):
		name_string_mid_str = str(args.num_kernels) + '_resnetrank_' + name_string_mid_str
		
	
	return name_string_mid_str

def assign_part_dataset(dataset, user_list=[]):
	
	if (args.model_name == 'inception'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(299),
			transforms.CenterCrop(299),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		transform_test = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(299),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		target_transform = transforms.ToTensor()
		
	elif (args.model_name == 'resnet50' or args.model_name == 'alexnet_large'
	      or args.model_name == 'densenet121'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		transform_test = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		target_transform = transforms.ToTensor()
		
	elif (args.model_name == 'mobilenetv3' or args.model_name == 'skin_special'):

		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		transform_test = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		target_transform = transforms.ToTensor()
	
	elif (dataset.dataset_name == 'mnist' or dataset.dataset_name == 'fashion_mnist' or (
			'celeb' in dataset.dataset_name) or dataset.dataset_name == 'retina' or dataset.dataset_name == 'medical_mnist'
	      or dataset.dataset_name == 'chest' or dataset.dataset_name == 'tb' or dataset.dataset_name == 'skin'
	      or dataset.dataset_name == 'kidney' or dataset.dataset_name == 'covid'):
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(64),
			#transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		])
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	elif (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
		
	elif (dataset.dataset_name == 'cifar10' or dataset.dataset_name == 'cifar100'):
		
		if (args.data_aug == 0): # center crop + horizontal clip
			transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif (args.data_aug == 1): # center crop
			transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(32),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif (args.data_aug == 2): # random crop
			transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif (args.data_aug == 3): # random crop + horizontal flip
			transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif (args.data_aug == 4): # random crop + horizontal flip + vertical flip
			transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			
		transform_test = transforms.Compose([
			transforms.ToTensor(),  ### totensor will perform the divide by 255 op
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		target_transform = transforms.ToTensor()
	
	
	num_users = len(user_list)
	
	### generating train / test indices for each user
	
	## for CIFAR-10 / CIFAR-100, we have 50000/10000 train/test data,
	## then each user should share the test data and we need to split the training data
	
	## for purchase and texas, we have enough data
	
	## the maximum number of user we can support is 10
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	
	assigned_index = []
	
	### for # of instance exp. each client has diff # of instance.
	#if (args.unequal):
	#	training_set_size_list = [400, 1200, 2000, 2800, 3600, 4400, 5200, 6000, 6800, 7600]
	
	for i in range(num_users):
		
		#if (args.unequal):
		#	training_set_size = training_set_size_list[i]
		#else:
		
		training_set_size = args.target_data_size
		
		this_user = user_list[i]
		this_user.target_transform = target_transform
		this_user.train_transform = transform_train
		this_user.test_transform = transform_test
		
		if (i == 0):
			print (this_user.train_transform)
		
		this_user_index = np.random.choice(len(index_left), training_set_size, replace=False)
		this_user_train_index = index_left[this_user_index]
		new_index_left = np.setdiff1d(np.arange(len(index_left)), this_user_index)
		index_left = index_left[new_index_left]
		
		this_user.train_data = dataset.train_data[this_user_train_index]
		this_user.train_label = dataset.train_label[this_user_train_index]
		
		#print(f"user {i} has classes:{np.bincount(dataset.train_label[this_user_train_index])}")
		# print (np.bincount(this_user.train_label))
		
		this_user.class_weight = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (
				len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label) + 1))
		
		# this_user.class_weight = np.ones((len(np.unique(this_user.train_label)))) * training_set_size / (len(np.unique(this_user.train_label)) * (np.bincount(this_user.train_label)))
		#print("class weight:", this_user.class_weight)
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
		### take a fraction of training data to be used as MI evaluation data (this is the member part of evaluation data)
		# when active attacker is not activated, here the eval_data_size == target_train_size
		# when active attacker is activated, here the eval_data_size == 100 or 50
		### if we are going to do targeted class GA mislabeling, then we need to make sure eval set in a specific class
		
		if (args.active_attacker_mislabel == 1 and args.mislabeling_target_class != -1):
			targeted_class = args.mislabeling_target_class
			this_class_index_in_train = np.arange(len(this_user.train_label))[this_user.train_label == targeted_class]
			eval_data_index = np.random.choice(this_class_index_in_train, args.eval_data_size, replace=False)
			#print(f"mislabel active attack eval data size {len(eval_data_index)}")
		else:
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
	
	### we select some data as validation set
	validation_data_index = np.random.choice(index_left, len(index_left),
	                                         replace=True)  ### this should be false, but just for the sake of # of user exp
	validation_data = dataset.train_data[validation_data_index]
	validation_label = dataset.train_label[validation_data_index]
	dataset.remaining_index = np.setdiff1d(index_left, validation_data_index)
	
	for user_idx in range(num_users):
		this_user = user_list[user_idx]
		this_user.eval_validation_data = validation_data
		this_user.eval_validation_label = validation_label
		### processing validation set for MMD defense
		
		### sort the validation data according to the class index
		sorted_index = np.argsort(this_user.eval_validation_label)
		this_user.eval_validation_data = this_user.eval_validation_data[sorted_index]
		this_user.eval_validation_label = this_user.eval_validation_label[sorted_index]
		
		### create an index list for starting index of each class
		this_user.starting_index = []
		# print ("starting index",self.starting_index)
		for i in np.unique(this_user.eval_validation_label):
			for j in range(len(this_user.eval_validation_label)):
				if (this_user.eval_validation_label[j] == i):
					this_user.starting_index.append(j)
					break
					
		this_user.validation_dataset = part_pytorch_dataset(validation_data, validation_label, train=False,
		                                                    transform=transform_test,
		                                                    target_transform=target_transform)
		this_user.validation_data_loader = torch.utils.data.DataLoader(this_user.validation_dataset,
		                                                               batch_size=args.target_batch_size, shuffle=False,
		                                                               num_workers=1)
		
		### create a validation-base-loader, to create a non member distribution, specifically for passive case
		validation_base_index = np.random.choice(len(validation_label),min(args.validation_set_size,len(validation_label)),replace=False)
		args.validation_set_size = len(validation_base_index)
		this_user.validation_base_data = validation_data[validation_base_index]
		this_user.validation_base_label = validation_label[validation_base_index]
		this_user.validation_base_dataset = part_pytorch_dataset(this_user.validation_base_data, this_user.validation_base_label, train=False,
		                                                    transform=transform_test,
		                                                    target_transform=target_transform)
		this_user.validation_base_data_loader = torch.utils.data.DataLoader(this_user.validation_base_dataset,
		                                                               batch_size=args.target_batch_size, shuffle=False,
		                                                               num_workers=1)
		
	#for user_idx in range(len(user_list)):
	#	np.save(f'./checkpoints/server_{args.dataset}_{args.model_name}_{args.target_data_size}_train_data_{user_idx}.npy',user_list[user_idx].train_data)
	#	np.save(f'./checkpoints/server_{args.dataset}_{args.model_name}_{args.target_data_size}_train_label_{user_idx}.npy',user_list[user_idx].train_label)
	#	np.save(f'./checkpoints/server_{args.dataset}_{args.model_name}_{args.target_data_size}_test_data_{user_idx}.npy',user_list[user_idx].test_data)
	#	np.save(f'./checkpoints/server_{args.dataset}_{args.model_name}_{args.target_data_size}_test_label_{user_idx}.npy',user_list[user_idx].test_label)
	

def run_blackbox_attacks(user_list, target_model, num_classes, output_file):
	# num_classes = 10 if (args.dataset == 'cifar10' or args.dataset == 'mnist') else 100
	acc = 0
	for user_idx in range(len(user_list)):
		black_ref = blackbox_attack(args.eval_data_size, 'global_prob', num_classes=num_classes)
		total_confidences, total_classes, total_labels = black_ref.get_attack_input(target_model, user_list[user_idx])
		acc += black_ref.attack(total_confidences=total_confidences, total_classes=total_classes,
		                        total_labels=total_labels, output_file=output_file)  ### labels here is the true label
	return acc / len(user_list)


def run_multi_party_attacks(user_list, target_model, epoch, user_update_list, user_model_list, ori_model_weight_dict,
                            server_attacker=False, attack_loader_list=[], comm_round_idx=0,best_layer=None):
	naming_str = get_naming_mid_str() + str(comm_round_idx) + '_' + str(epoch + 1) + '_' + str(
		args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
		args.model_name) + '.npy'
	
	save_path_prefix = './new_expdata/'
	if (args.server_name == 'ramos'):
		save_path_prefix = '/homes/li2829/scratch/fed/fed_expdata/'
		
	#print (f"attacker role {server_attacker}")
	
	### for a few cases like inception, we don't need the whole model, just the last two layers..
	if (args.model_name == 'inception' or args.model_name == 'resnet50'):
		
		#for idx,l in enumerate(target_model.layers()):
		#	print (idx,l)
		#print (target_model)
		#for idx,p_in_model in enumerate(target_model.parameters()):
		#	print (idx,p_in_model.size(),p_in_model.requires_grad)
		#print (len(user_update_list[0]))
		#new_user_update_list = []
		#for user_model in user_update_list:
		#	this_user_update_list = []
		#	for p,p_in_model in zip(user_model,target_model.parameters()):
		#		print (p.size(),p_in_model.size())
		#		if (p_in_model.requires_grad):
		#			this_user_update_list.append(p)
		#			print ("YES")
		#
		#new_user_update_list.append(this_user_update_list)
		#print (f"filtered model update list length {len(this_user_update_list)}")
		# just two param. weight / bias
		user_update_list = [user_update_list[i][-2:]  for i in range(len(user_update_list))]
		#exit(0)
	#print(naming_str)
	
	#print (f"user update list length{len(user_update_list)}, {len(user_update_list[0])}")
	
	if (server_attacker):
		
		all_info, all_label = multi_party_member_attack(user_list, target_model, batch_size=args.target_batch_size,
		                                                user_update_list=user_update_list,
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list, user_total_instance=args.num_step * args.target_batch_size,
		                                                max_instance_per_batch=args.max_instance_per_batch,best_layer=best_layer,test_rank=args.test_rank,
		                                                whole_nn=args.whole_nn)
		np.save(save_path_prefix + 'all_info_multi_party_member_attack_' + naming_str, all_info)
		np.save(save_path_prefix + 'all_label_multi_party_member_attack_' + naming_str, all_label)
		
		print (all_info.shape,all_label.shape)

		loss_info, loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                       batch_size=args.target_batch_size,
		                                                       get_gradient_func=get_gradient,
		                                                       user_model_list=user_model_list,
		                                                       attack_loader_list=attack_loader_list,
		                                                       max_instance_per_batch=args.max_instance_per_batch,
		                                                       model_name=args.model_name)
		np.save(save_path_prefix + 'loss_info_multi_party_member_attack_' + naming_str, loss_info)
		np.save(save_path_prefix + 'loss_label_multi_party_member_attack_' + naming_str, loss_label)
		
		
		if (not args.active_attacker):
			valid_info,valid_label = multi_party_member_attack_valid(user_list, target_model, batch_size=args.target_batch_size,
		                                                user_update_list=user_update_list,
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list, user_total_instance=args.num_step * args.target_batch_size,
		                                                max_instance_per_batch=args.max_instance_per_batch,best_layer=best_layer,test_rank=args.test_rank,
			                                            whole_nn=args.whole_nn)
			np.save(save_path_prefix + 'valid_all_info_multi_party_member_attack_' + naming_str, valid_info)
			np.save(save_path_prefix + 'valid_all_label_multi_party_member_attack_' + naming_str, valid_label)
			
			valid_loss_info, valid_loss_label = multi_party_member_loss_attack_valid(user_list, target_model,
			                                                       batch_size=args.target_batch_size,
			                                                       user_update_list=user_update_list,
			                                                       get_gradient_func=get_gradient,
			                                                       user_model_list=user_model_list,
			                                                       attack_loader_list=attack_loader_list,
			                                                       max_instance_per_batch=args.max_instance_per_batch,
		                                                           model_name=args.model_name)
			np.save(save_path_prefix + 'valid_loss_info_multi_party_member_attack_' + naming_str, valid_loss_info)
			np.save(save_path_prefix + 'valid_loss_label_multi_party_member_attack_' + naming_str, valid_loss_label)
		
		'''
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=user_update_list,
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[ori_model_weight_dict],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)

		weights_after_avg = average_weights(user_model_list)
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=user_update_list,
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[weights_after_avg],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		
		print("server adversary - attack finished")
		
		if (len(user_list) == 1):
			print (np.squeeze(target_loss_info)[:50])
		
		return
	'''
	if (not server_attacker):
		### if the client is the adversary, we need to change the user_update_list to make it the sum of updates from all other parties
		### here we assume that the last party is the adversary
		sum_user_update_list = []
		for param in user_update_list[0]:
			sum_user_update_list.append(torch.zeros_like(param))
		
		for user_idx in range(len(user_list) - 1):
			for idx, param in enumerate(user_update_list[user_idx]):
				sum_user_update_list[idx] = sum_user_update_list[idx] + param
		
		for param in sum_user_update_list:
			param = param / (len(user_list) - 1)
		
		### for the loss attack, the available model is the model updated with sum of updates from all other parties
		temp_sum_weights = average_weights(user_model_list[:-1])
		
		all_info, all_label = multi_party_member_attack(user_list[:-1], target_model, batch_size=args.target_batch_size,
		                                                user_update_list=[sum_user_update_list],
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list,
		                                                user_total_instance=args.num_step * args.target_batch_size * (len(user_list) - 1),
		                                                max_instance_per_batch=args.max_instance_per_batch,best_layer=best_layer,whole_nn=args.whole_nn)
		np.save('./new_expdata/all_info_multi_party_member_attack_' + naming_str, all_info)
		np.save('./new_expdata/all_label_multi_party_member_attack_' + naming_str, all_label)
		
		#print ('./new_expdata/all_info_multi_party_member_attack_' + naming_str)
		
		loss_info, loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                       batch_size=args.target_batch_size,
		                                                       user_update_list=[sum_user_update_list],
		                                                       get_gradient_func=get_gradient,
		                                                       user_model_list=[temp_sum_weights],
		                                                       attack_loader_list=attack_loader_list,
		                                                       max_instance_per_batch=args.max_instance_per_batch,
		                                                       model_name=args.model_name)
		np.save('./new_expdata/loss_info_multi_party_member_attack_' + naming_str, loss_info)
		np.save('./new_expdata/loss_label_multi_party_member_attack_' + naming_str, loss_label)
		
		if (not args.active_attacker):
			valid_info, valid_label = multi_party_member_attack_valid(user_list[:-1], target_model, batch_size=args.target_batch_size,
			                                                          user_update_list=[sum_user_update_list],
			                                                          get_gradient_func=get_gradient,
			                                                          attack_loader_list=attack_loader_list,
			                                                          user_total_instance=args.num_step * args.target_batch_size * (len(user_list) - 1),
			                                                          max_instance_per_batch=args.max_instance_per_batch, best_layer=best_layer, test_rank=args.test_rank,
			                                                          whole_nn=args.whole_nn)
			np.save(save_path_prefix + 'valid_all_info_multi_party_member_attack_' + naming_str, valid_info)
			np.save(save_path_prefix + 'valid_all_label_multi_party_member_attack_' + naming_str, valid_label)
			
			valid_loss_info, valid_loss_label = multi_party_member_loss_attack_valid(user_list[:-1], target_model,
			                                                                         batch_size=args.target_batch_size,
			                                                                         user_update_list=[sum_user_update_list],
			                                                                         get_gradient_func=get_gradient,
			                                                                         user_model_list=[temp_sum_weights],
			                                                                         attack_loader_list=attack_loader_list,
			                                                                         max_instance_per_batch=args.max_instance_per_batch,
			                                                                         model_name=args.model_name)
			np.save(save_path_prefix + 'valid_loss_info_multi_party_member_attack_' + naming_str, valid_loss_info)
			np.save(save_path_prefix + 'valid_loss_label_multi_party_member_attack_' + naming_str, valid_loss_label)
		'''
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=[sum_user_update_list],
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[ori_model_weight_dict],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_before_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_before_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		
		weights_after_avg = average_weights(user_model_list)
		target_loss_info, target_loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                                     batch_size=args.target_batch_size,
		                                                                     user_update_list=[sum_user_update_list],
		                                                                     get_gradient_func=get_gradient,
		                                                                     user_model_list=[weights_after_avg],
		                                                                     attack_loader_list=attack_loader_list,
		                                                                     max_instance_per_batch=args.max_instance_per_batch)
		np.save('./new_expdata/target_model_after_loss_info_multi_party_member_attack_' + naming_str, target_loss_info)
		np.save('./new_expdata/target_model_after_loss_label_multi_party_member_attack_' + naming_str, target_loss_label)
		'''
		#print("client adversary - attack finished")
		return

	
def merlin_attack_count(data,label,model,noise_magnitude=0.01):
	### all the hyperparameters are the same as the original paper
	### for normalized images, the noise magnitude is 0.01
	### for unnormalized images, we set the noise magnitude to 0.01*255
	### for purchase, we use 0.01
	repeat_times = 100
	counts = []
	
	if (args.dataset == 'cifar100' or args.dataset == 'cifar10' or args.dataset == 'purchase' or args.dataset == 'texas'
		or args.model_name == 'resnet50' or args.model_name == 'inception'):
		noise_magnitude = 0.01
	else:
		noise_magnitude = 0.01*255/2
	
	pred = F.softmax(model(data),dim=1)
	probs = np.array([pred[i,label[i]].detach().item() for i in range(len(label))])
	
	### for each instance in data, we randomly generate noise for 100 rounds and count the number of times that new loss is larger
	for img_idx in range(len(data)):
		this_img = data[img_idx]
		this_label = label[img_idx]
		### repeat this img for 100 times
		stacked_img = torch.stack([this_img for _ in range(repeat_times)])
		stacked_std = torch.ones_like(stacked_img)*noise_magnitude
		random_noise = torch.normal(mean=0,std=stacked_std)
		noisy_img = stacked_img + random_noise
		#print (stacked_img.size(),stacked_std.size(),random_noise.size(),noisy_img.size())
		noisy_pred = F.softmax(model(noisy_img),dim=1)
		noisy_probs =  np.array([noisy_pred[i,this_label].detach().item() for i in range(repeat_times)])
		### larger loss means smaller probs, so we count the number of times for smaller probs.
		this_count = len(np.arange(len(noisy_probs))[noisy_probs<probs[img_idx]])
		#print (noisy_probs)
		#print (probs[img_idx])
		#print (this_count)
		counts.append(this_count)
		
	return counts

def modified_entropy(x,y):
	## shape of x: [# of data, # of classes]. x is a batch of prediction
	modified_entropy = []
	for i in range(len(y)):
		this_pred = x[i]
		this_label = y[i]
		this_modified_entropy = 0
		for j in range(len(this_pred)):
			if (j == this_label):
				this_modified_entropy = this_modified_entropy - (1-this_pred[this_label]) * torch.log(this_pred[this_label])
			else:
				this_modified_entropy = this_modified_entropy - (this_pred[j]) * torch.log (1 - this_pred[j])
		modified_entropy.append(this_modified_entropy.detach().item())
		
	return np.nan_to_num(np.array(modified_entropy)*-1,posinf=1e9)
	

def get_blackbox_auc(user_list,target_model):
	
	yeom_auc = []
	yeom_tpr = []
	## yeom's attack
	for idx in range(len(user_list)):
		member_probs = []
		nonmember_probs = []
		for (images,labels,_) in user_list[idx].train_eval_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images),dim=1)
			#probs = preds[:,labels]
			probs = np.array([preds[i,labels[i]].detach().item() for i in range(len(labels))])
			#print (idx,probs.size())
			member_probs.append(probs)
	
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images),dim=1)
			#print (preds[0],torch.sum(preds[0]))
			#probs = preds[:, labels]
			probs = np.array([preds[i,labels[i]].detach().item() for i in range(len(labels))])
			#print (preds.size(),labels.size(), probs.shape)
			nonmember_probs.append(probs)
		
		member_probs = np.concatenate(member_probs).flatten()
		nonmember_probs = np.concatenate(nonmember_probs).flatten()
		min_len = min(len(member_probs),len(nonmember_probs))
		min_len = min(min_len,args.eval_data_size)
		#print (f"min len {min_len},member len{len(member_probs)}, nonmember len {len(nonmember_probs)}")
		member_index = np.random.choice(len(member_probs),min_len,replace=False)
		nonmember_index = np.random.choice(len(nonmember_probs),min_len,replace=False)
		#print (len(member_index),len(nonmember_index))
		probs = np.concatenate((member_probs[member_index],nonmember_probs[nonmember_index]),axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)),np.zeros((min_len))),axis=0).astype(np.int64).flatten()
	
		#print (probs.shape,labels.shape)

		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(labels,probs)
		#print (f"BLACKBOX LOSS AUC {auc_score}")

		from sklearn.metrics import roc_auc_score, roc_curve
		fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
	
		return_tpr = get_tpr(pred=probs,label=labels)
		#print (f"FPR {10/min_len}, TPR {return_tpr}")
		
		yeom_auc.append(auc_score)
		yeom_tpr.append(return_tpr)
		
	print(f"yeom attack: avg auc {np.average(np.array(yeom_auc))}, avg tpr {np.average(np.array(yeom_tpr))} at fpr {10/min_len}")
	print(f"auc std : {np.std(np.array(yeom_auc))}, tpr std :{np.std(np.array(yeom_tpr))}")
	
	#### here is the merlin attack
	merlin_auc = []
	merlin_tpr = []
	for idx in range(len(user_list)):
		member_counts = []
		nonmember_counts = []
		for (images, labels, _) in user_list[idx].train_eval_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			counts = merlin_attack_count(images,labels,target_model,noise_magnitude=0.01)
			member_counts.append(counts)
	
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			counts = merlin_attack_count(images,labels,target_model,noise_magnitude=0.01)
			nonmember_counts.append(counts)
	
		member_counts = np.concatenate(member_counts).flatten()
		nonmember_counts = np.concatenate(nonmember_counts).flatten()
		#print (member_counts.shape)
		min_len = min(len(member_counts), len(nonmember_counts))
		min_len = min(min_len, args.eval_data_size)
		#print(f"min len {min_len},member len{len(member_counts)}, nonmember len {len(nonmember_counts)}")
		member_index = np.random.choice(len(member_counts), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_counts), min_len, replace=False)
		# print (len(member_index),len(nonmember_index))
		counts = np.concatenate((member_counts[member_index], nonmember_counts[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
	
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(labels, counts)
		from sklearn.metrics import roc_auc_score, roc_curve
		fpr, tpr, thresholds = roc_curve(labels, counts, pos_label=1)
		return_tpr = get_tpr(pred=counts, label=labels)
		#print(f"FPR {10 / min_len}, TPR {return_tpr}")
		merlin_auc.append(auc_score)
		merlin_tpr.append(return_tpr)
		
		#print (member_counts)
		#print (nonmember_counts)
		if (idx == 0):
			print (np.bincount(member_counts))
			print (np.bincount(nonmember_counts))
		
	print (merlin_auc)
	print (merlin_tpr)
	
	print(f"merlin attack: avg auc {np.average(np.array(merlin_auc))}, avg tpr {np.average(np.array(merlin_tpr))} at fpr {10 / min_len}")
	print (f"auc std : {np.std(np.array(merlin_auc))}, tpr std :{np.std(np.array(merlin_tpr))}")
	
	### here is the modified entropy attack
	song_auc = []
	song_tpr = []
	## song's attack
	for idx in range(len(user_list)):
		member_probs = []
		nonmember_probs = []
		for (images, labels, _) in user_list[idx].train_eval_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			# probs = preds[:,labels]
			probs = modified_entropy(preds,labels)
			# print (idx,probs.size())
			member_probs.append(probs)
		
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			# print (preds[0],torch.sum(preds[0]))
			# probs = preds[:, labels]
			probs = modified_entropy(preds,labels)
			# print (preds.size(),labels.size(), probs.shape)
			nonmember_probs.append(probs)
		
		member_probs = np.concatenate(member_probs).flatten()
		nonmember_probs = np.concatenate(nonmember_probs).flatten()
		min_len = min(len(member_probs), len(nonmember_probs))
		min_len = min(min_len, args.eval_data_size)
		#print(f"min len {min_len},member len{len(member_probs)}, nonmember len {len(nonmember_probs)}")
		member_index = np.random.choice(len(member_probs), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_probs), min_len, replace=False)
		# print (len(member_index),len(nonmember_index))
		probs = np.concatenate((member_probs[member_index], nonmember_probs[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
		
		# print (probs.shape,labels.shape)
		
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(labels, probs)
		# print (f"BLACKBOX LOSS AUC {auc_score}")
		
		from sklearn.metrics import roc_auc_score, roc_curve
		fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
		
		return_tpr = get_tpr(pred=probs, label=labels)
		# print (f"FPR {10/min_len}, TPR {return_tpr}")
		
		song_auc.append(auc_score)
		song_tpr.append(return_tpr)
	
	print(f"modified entropy attack: avg auc {np.average(np.array(song_auc))}, avg tpr {np.average(np.array(song_tpr))} at fpr {10 / min_len}")
	print(f"auc std : {np.std(np.array(song_auc))}, tpr std :{np.std(np.array(song_tpr))}")


def train_models(user_list, target_model, learning_rate, decay, epochs, class_weights=None,target_dataset=None):
	num_users = len(user_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.to(device)
	momentum = 0.9
	
	#if (epochs == 100):
	#	save_model_checkpoint_epoch_list = np.array([60,70,80,90,100])-1
	#elif (epochs == 150):
	#	save_model_checkpoint_epoch_list = np.array([110,120,130,140,150])-1
	#elif (epochs == 300):
	#	save_model_checkpoint_epoch_list = np.array([100, 150, 200, 250, 300])-1
	#elif (epochs == 500):
	#	save_model_checkpoint_epoch_list = np.array([300,350,400,450,500])-1
	
	grad_norm_results = []
	all_acc_results = []
	all_loss_results = []
	all_active_results = []
	all_multiplier_results = []
	lr_scheduling_count = 0
	acc_lambda = 0
	
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		if (args.dataset == 'purchase' or args.dataset == 'texas' or args.model_name == 'resnet50' or args.model_name == 'inception'): #
			if (args.signsgd):
				this_optim = SignedAdam(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, weight_decay=decay)
			else:
				this_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, weight_decay=decay)
		else:
			if (args.signsgd):
				this_optim = Signum(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
			else:
				this_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, user_list[user_idx].model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
		
		user_list[user_idx].optim = this_optim
		
		if (args.dpsgd):
			### adding dp components
			#print ("ADD DP TO USER MODEL")
			user_list[user_idx].privacy_engine = PrivacyEngine()
			user_list[user_idx].model, user_list[user_idx].optim, user_list[user_idx].train_data_loader  = user_list[user_idx].privacy_engine.make_private(
				module=user_list[user_idx].model,
				optimizer=user_list[user_idx].optim,
				data_loader=user_list[user_idx].train_data_loader,
				# sample_rate=(args.target_batch_size / args.target_data_size),
				# alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),  ### params for renyi dp
				noise_multiplier=args.noise_scale,  ### sigma
				max_grad_norm=args.grad_norm)  ### this is from dp-sgd paper)
	# privacy_engine.attach(user_list[user_idx].optim)

	if (args.active_attacker):
		attacker_model_copy = copy.deepcopy(target_model)
	
	### for dpsgd case.. just to make sure the name of parameters for target model is the same as other private models,
	if (args.dpsgd):
		print ("DPSGD ACTIVATED")
		target_model_privacy_engine = PrivacyEngine()
		if (args.dataset == 'purchase' or args.dataset == 'texas' or args.model_name == 'resnet50' or args.model_name == 'inception' or args.model_name == 'mobilenetv3'):
			target_model_optim = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=decay)
		else:
			target_model_optim = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
		 
		train = part_pytorch_dataset(target_dataset.train_data, target_dataset.train_label, train=True, transform=None,target_transform=None)
		target_model_train_loader =  torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,shuffle=True, num_workers=1)
		target_model, target_model_optim, target_model_train_loader  = target_model_privacy_engine.make_private(
				module=target_model,
				optimizer=target_model_optim,
				data_loader=target_model_train_loader,
				# sample_rate=(args.target_batch_size / args.target_data_size),
				# alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),  ### params for renyi dp
				noise_multiplier=args.noise_scale,  ### sigma
				max_grad_norm=args.grad_norm)  ### this is from dp-sgd paper)
		
	### start training
	for epoch in tqdm(range(epochs)):
		if (args.repartition):
			repartition_dataset(user_list)
		
		## calculate avg cosine similarity
		#file_name = './checkpoints/server_cifar100_alexnet_4000_250.pt'
		#checkpoint = torch.load(file_name)
		#target_model.load_state_dict(checkpoint['model_state_dict'])
		#cos_result = calculate_avg_cosine_similarity(user_list, target_model,get_gradient_func=get_gradient)
		#np.save('avg_cos_result_full_model.npy',cos_result)
		#print (f"avg cos {np.average(cos_result)}, std cos {np.std(cos_result)}")
		#exit(0)
		
		## calculate gradient for a set of samples...
		#file_name = './checkpoints/server_cifar100_alexnet_4000_250.pt'
		#checkpoint = torch.load(file_name)
		#target_model.load_state_dict(checkpoint['model_state_dict'])
		#member_images = np.transpose(user_list[0].train_data[:100],(0,3,1,2))
		#print (member_images.shape)
		#member_images = torch.from_numpy(member_images).type(torch.float32).cuda()
		#member_labels = torch.from_numpy(user_list[0].train_label[:100]).cuda()
		#nonmember_images = np.transpose(user_list[0].test_data[:100],(0,3,1,2))
		#nonmember_images = torch.from_numpy(nonmember_images).type(torch.float32).cuda()
		#nonmember_labels = torch.from_numpy(user_list[0].test_label[:100]).cuda()
		#member_gradients = get_gradient(target_model,member_images,member_labels)
		#nonmember_gradients = get_gradient(target_model,nonmember_images,nonmember_labels)
		#member_gradients = np.concatenate([ np.reshape(x.cpu().numpy(),(100,-1)) for x in member_gradients],axis=1)
		#nonmember_gradients = np.concatenate([np.reshape(x.cpu().numpy(),(100,-1)) for x in nonmember_gradients],axis=1)
		#print (member_gradients.shape)
		#np.save('member_gradients.npy',member_gradients)
		#np.save('nonmember_gradients.npy',nonmember_gradients)
		#exit(0)
		
		#if (epoch in save_model_checkpoint_epoch_list):
			### save model checkpoints for nasr's attack using user 0
		#	checkpoint_name = './checkpoints/server_' + args.dataset + '_' + args.model_name + '_' +  str(args.target_data_size) + '_' + str(epoch+1) + '.pt'
		#	torch.save({'model_state_dict': user_list[0].model.state_dict(), 'optimizer_state_dict': user_list[0].optim.state_dict()}, checkpoint_name)
		#	torch.save({'model_state_dict': target_model.state_dict(), 'optimizer_state_dict': user_list[0].optim.state_dict()}, checkpoint_name)
		#	print (f"save model checkpoint {checkpoint_name}")
		
		ori_target_model_state_dict = target_model.state_dict()  ### should we copy this?
		## LR schedule
		if (epoch in args.schedule):
			learning_rate = learning_rate / 10
			print("new learning rate = %f" % (learning_rate))
			### set up new learning rate for each user
			for user_idx in range(num_users):
				for param_group in user_list[user_idx].optim.param_groups:
					param_group['lr'] = learning_rate
			
			lr_scheduling_count+=1
		
		## active attacker gradient ascent
		if (args.active_attacker):
			if (args.dpsgd):
				## for this case, we need to load the param one by one
				attacker_param = attacker_model_copy.state_dict()
				for (key1,val1),(key2,val2) in zip(attacker_param.items(),ori_target_model_state_dict.items()):
					#print (key1,key2,val1.size(),val2.size())
					#if (len(val1.size())==1):
					#	print (val1)
					#	print (val2)
					val1.data.copy_(val2.data)
					#if (len(val1.size())==1):
					#	print (val1)
					#	print (val2)
					
			else:
				attacker_model_copy.load_state_dict(ori_target_model_state_dict)
			
			if (args.dataset == 'purchase' or args.dataset == 'texas'):
				#attacker_optim = torch.optim.SGD(attacker_model_copy.parameters(), lr=learning_rate, momentum=momentum,weight_decay=decay)
				attacker_optim = torch.optim.Adam(attacker_model_copy.parameters(), lr=learning_rate, weight_decay=decay)
			else:
				attacker_optim = torch.optim.SGD(attacker_model_copy.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
			
			# gradient implementation
			# gradient_ascent_gradient, active_magnitude,active_loss = active_attacker_gradient_ascent(attacker_model_copy,attacker_optim,user_list,
			# local_epochs=args.local_epoch,batch_size=50,client_adversary=args.client_adversary,
			# lr_multiplier=args.active_attacker_lr_multiplier,param_search=args.active_attack_param_search)
			
			# optim implementation
			ga_adversary = 0.01 if (args.active_attacker_lr_multiplier>0) else -1 ### setting 0.01 for GA is to make sure there is no NAN
			gradient_ascent_weight, active_loss = active_attacker_gradient_ascent(attacker_model_copy, attacker_optim, user_list,
			                                                                      ga_adversary=ga_adversary,local_epochs=1)
			#### calculate magnitude
			active_magnitude = calculate_param_change_magnitude(ori_target_model_state_dict, gradient_ascent_weight)
		
		#### we need to know the # of steps, # of training set size to perform fed sgd
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		#print(f"comm round per epoch{comm_round_per_epoch}")
		
		for comm_round_idx in range(comm_round_per_epoch):
			local_weights = []
			attack_loader_list = []
			for idx in range(len(user_list)):
				# create a new dataloader
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				### defense
				if (args.mmd):
					_ = update_weights_mmd(current_model_weights=ori_target_model_state_dict,
					                       model=user_list[idx].model, optimizer=user_list[idx].optim,
					                       train_loader=user_list[idx].train_data_loader,
					                       test_loader=user_list[idx].test_data_loader,
					                       local_epochs=args.local_epochs, mixup=args.mixup,
					                       validation_set=user_list[idx].validation_dataset,
					                       loss_lambda=args.mmd_loss_lambda,
					                       train_loader_in_order=user_list[idx].train_loader_in_order,
					                       num_classes=len(np.unique(user_list[idx].train_label)),
					                       starting_index=user_list[idx].starting_index,
					                       class_weights=user_list[idx].class_weight)
				else:
					# print ("no defense applied")
					ban_list = update_weights(current_model_weights=ori_target_model_state_dict,
					                             model=user_list[idx].model,
					                             optimizer=user_list[idx].optim, train_loader=train_data_loader,
					                             local_epochs=args.local_epochs, mixup=args.mixup,
					                             num_step=args.num_step,
					                             class_weights=user_list[idx].class_weight, unequal=args.unequal,
					                             model_name=args.model_name) ### model name is specifically for inception..
					
					#attack_loader_list.append(user_list[idx].create_batch_attack_data_loader(ban_list,batch_size=args.max_instance_per_batch))
					user_list[idx].update_ban_list(ban_list)
					torch.cuda.empty_cache()
					
				local_weights.append((user_list[idx].model.state_dict()))
			
			#print ("show one model state dict")
			#for k,v in local_weights[0]:
			#	print (k,v.size())
			
			#print(f" TEST:num step{args.num_step}, epoch{epoch}, attacker_epoch_gap {args.attacker_epoch_gap},comm_round {comm_round_idx}, "
			#      f"comm_round_per_epoch {comm_round_per_epoch}")
		
			### batch level attack
			if (args.num_step == 1 and epoch % args.attacker_epoch_gap == 0): #  and epoch > 0
				num_batches = 1
				#print(f"epoch {epoch}, comm_round_idx {comm_round_idx}, num step {args.num_step}")
				user_update_list = get_user_update_list(ori_target_model_state_dict, local_weights,
				                                        learning_rate=learning_rate, num_batches=num_batches)
				# _, _ = get_train_test_acc(user_list, target_model)
				local_model = copy.deepcopy(target_model)
				run_multi_party_attacks(user_list, local_model, epoch, user_update_list,
				                        user_model_list=local_weights,
				                        server_attacker=1 - args.client_adversary,
				                        ori_model_weight_dict=ori_target_model_state_dict,
				                        attack_loader_list=[],
				                        comm_round_idx=comm_round_idx,best_layer=args.best_layer)
			# print (f"finish batch level attacks")
				del user_update_list,local_model
			
			### epoch level attack
			elif (epoch % args.attacker_epoch_gap == 0 and comm_round_idx + 1 == comm_round_per_epoch and epoch > 0):  # and epoch > 0 NEED TO RESTORE
				### this atack requires # of steps = # of batches
				num_batches = int(args.target_data_size / args.target_batch_size)
				if (args.num_step != num_batches):
					print("epoch attack: num steps not equal to num batches")
					exit(0)
				
				#print(f"EPOCH LEVEL ATTACK:epoch {epoch}, comm_round_idx {comm_round_idx}, num step {args.num_step}")
				
				user_update_list = get_user_update_list(ori_target_model_state_dict, local_weights, learning_rate=learning_rate, num_batches=num_batches)
				
				local_model = copy.deepcopy(target_model)
				run_multi_party_attacks(user_list, local_model, epoch, user_update_list,
				                        user_model_list=local_weights,
				                        server_attacker=1 - args.client_adversary,
				                        ori_model_weight_dict=ori_target_model_state_dict,
				                        attack_loader_list=[],
				                        comm_round_idx=comm_round_idx,best_layer=args.best_layer)
				#print(f"finish attacks for {epoch} epochs")
				del user_update_list,local_model
				
			### update global weights
			global_weights = average_weights(local_weights)
			del local_weights
			
			if (args.active_attacker):
				# active_attack_multiplier = -0.05 * (normal_magnitude*100/ (learning_rate*args.user_number*args.target_data_size)) / (active_magnitude * 50 / (2*args.eval_data_size*args.user_number))
				# active_attack_multiplier = args.active_attacker_lr_multiplier*-1
				# if (args.active_attacker_lr_multiplier > 0):
				#	ga_constant = -1
				#	active_attack_multiplier = active_attack_multiplier * ga_constant
				
				normal_magnitude = calculate_param_change_magnitude(ori_target_model_state_dict, global_weights) * 10
				normal_loss = get_train_loss(user_list, copy.deepcopy(target_model), ori_target_model_state_dict)
				avg_train_norm = (normal_magnitude.cpu().item() * 100) / (learning_rate * args.user_number * args.target_data_size)
				avg_attack_norm = active_magnitude.cpu().item() * 50 * 100 / (learning_rate * args.user_number * 2 * args.eval_data_size)
				#upper_bound = normal_magnitude / (active_magnitude*100)
				upper_bound = avg_train_norm/avg_attack_norm*(args.target_data_size/args.eval_data_size)/2 * args.user_number
				#print (f"upper bound {upper_bound}")
				# if (args.active_attacker_lr_multiplier>0): ### GA case
				#	active_attack_multiplier = 1 * normal_loss / active_loss *avg_train_norm / avg_attack_norm ### 1 is required for GA.
				# else:### GD case
				#	active_attack_multiplier = -1 * active_loss / active_loss*avg_train_norm / avg_attack_norm  ### -1 is required for GD. We add gradient to model.
				# print(f"ratio {active_attack_multiplier} ")
				
				#'''
				### we need to perform a param search here.
				if (args.active_attacker_lr_multiplier<0):
					alpha,log = gd_active_attacker_param_search(user_list, copy.deepcopy(target_model), ori_target_model_state_dict, gradient_ascent_weight, global_weights,scheduling=lr_scheduling_count,upper_bound=upper_bound)
					if (alpha < 0):
						alpha = 0
					active_attack_multiplier = alpha
				else:
					#alpha, log = ga_active_attacker_param_search(user_list, copy.deepcopy(target_model), ori_target_model_state_dict, gradient_ascent_weight, global_weights)
					#active_attack_multiplier = alpha
					active_attack_multiplier = args.active_attacker_lr_multiplier*100
					#active_attack_multiplier = avg_train_norm / avg_attack_norm * args.active_attacker_lr_multiplier * 100
					#active_attack_multiplier = normal_magnitude / (active_magnitude*100) * args.active_attacker_lr_multiplier * 100
					log = []
				#'''
				
				### for previous comparison only
				#active_attack_multiplier = args.active_attacker_lr_multiplier* -1
				#log = []
				
				all_active_results.append(log)
				all_multiplier_results.append(active_attack_multiplier)
				#print(f"this epoch {epoch}, multiplier {active_attack_multiplier}")
				# all_active_results.append((alpha_list,this_epoch_valid_loss))
				
				new_weight_dict = {}
				# optim implementation
				for (key1, val1), (key2, val2), (key3, val3) in zip(ori_target_model_state_dict.items(), gradient_ascent_weight.items(), global_weights.items()):
					if(active_attack_multiplier!=0):
						new_weight_dict[key1] = val3 + (val2 - val1) * active_attack_multiplier
					else:
						new_weight_dict[key1] = val3 ## this is to deal with numerical problems when gradient ascent weight is NaN (only for GA)
				
				# gradient implementation
				# for (key1,val1),(key2,val2),(key3,val3) in zip(ori_target_model_state_dict.items(),gradient_ascent_gradient.items(),global_weights.items()):
				# active_attack_multiplier = torch.nan_to_num(torch.norm(torch.flatten(val1-val3),p=1)/torch.norm(torch.flatten(val2),p=1))
				# new_weight_dict[key2] = val3 + val2*active_attack_multiplier
				
				# optim implementation
				cos_result = calculate_param_cosine_similarity(gradient_ascent_weight, global_weights, ori_target_model_state_dict)
				
				# gradient implementation
				# cos_result = -1 * calculate_param_cosine_similarity(copy.deepcopy(gradient_ascent_gradient),copy.deepcopy(global_weights),copy.deepcopy(ori_target_model_state_dict))
				
				grad_norm_results.append((avg_train_norm, avg_attack_norm, normal_loss, active_loss, cos_result))
				
				#print (f"overall train norm {normal_magnitude}, overall attack norm {active_magnitude}, ratio {normal_magnitude/active_magnitude}")
				#print (f"avg train norm {avg_train_norm}, avg attack norm {avg_attack_norm}, ratio {avg_train_norm/avg_attack_norm}, "
				#       f"multiplier {active_attack_multiplier}, normal loss {normal_loss}, active loss {active_loss}")
				#print (f"good lambda {avg_attack_norm/avg_train_norm*active_attack_multiplier}")
				#print (f" upper bound {avg_train_norm/avg_attack_norm*(args.target_data_size/args.eval_data_size)/2}")
				acc_lambda += avg_attack_norm/avg_train_norm*active_attack_multiplier
				
				target_model.load_state_dict(new_weight_dict)
				
				#print(get_train_test_acc(user_list, target_model))
			
			else:
				target_model.load_state_dict(global_weights)
			
			if (args.track_loss):
				#print ("MODEL AFTER ATTACK")
				train_acc, test_acc, val_acc, train_loss, test_loss, val_loss = get_train_test_acc(user_list,copy.deepcopy(target_model),print_option=True,
				                                                                                   return_loss=True,return_validation_result=True)
				local_model = copy.deepcopy(target_model)
				local_model.load_state_dict(global_weights)
				old_val_acc = val_acc
				#print ("MODEL BEFORE ATTACK")
				#train_acc, test_acc, val_acc, train_loss, test_loss, val_loss = get_train_test_acc(user_list, copy.deepcopy(local_model), print_option=True,
				#                                                                                   return_loss=True,return_validation_result=True)
				#new_val_acc = val_acc
				#print (f"VAL ACC DROP {(1-old_val_acc/new_val_acc)*100} PERCENT")
				#all_loss_results.append((train_loss, val_loss, test_loss))
				#all_acc_results.append((train_acc, test_acc, val_acc))
				#print (f"validation loss {val_loss}")
			
	if (args.track_loss):
		loss_str = './new_expdata/track_loss_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
			args.target_data_size) + '_' + str(args.model_name) + '.npy'
		all_loss_results = np.array(all_loss_results)
		#np.save(loss_str, all_loss_results)
		
		acc_str = './new_expdata/track_acc_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
			args.target_data_size) + '_' + str(args.model_name) + '.npy'
		all_acc_results = np.array(all_acc_results)
		#np.save(acc_str, all_acc_results)
	
	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"train acc {train_acc},test acc {test_acc}")
	
	return target_model, train_acc, test_acc


def get_gradient(target_model, data, label):  ### using gpu 1
	import time
	from opacus import GradSampleModule
	#device = torch.device("cuda",1) # use extra gpu if time is more important so we can process larger batches
	device = torch.device("cuda",0)
	criterion = nn.CrossEntropyLoss().to(device)
	target_model = target_model.to(device)
	if (args.dpsgd==0):
		local_model = copy.deepcopy(target_model)
		local_model = GradSampleModule(local_model)
		local_model.zero_grad()
	else:
		### one issue is, deepcopy cannot copy p._forward_counter for some reasons, so we need to add _forward_counter and grad_sample by hand..
		local_model = target_model
		for _,p in local_model.named_parameters():
			p._forward_counter = 0
			p.grad_sample = None
		
	#for name, param in local_model.named_parameters():
	#	print(name, param.size())
	
	#print (data.size(),label.size())
	#print (type(target_model))
	'''
	grad_list = []
	### one by one implementation
	start_time = time.time()
	for this_data,this_label in zip(data,label):
		this_data = this_data.to(device)
		this_label = this_label.to(device)
		this_data = torch.unsqueeze(this_data,dim=0)
		this_label = torch.unsqueeze(this_label,dim=0)
		prediction = local_model(this_data)
		loss = criterion(prediction, this_label)
		loss.backward()
		this_grad = copy.deepcopy([param.grad.detach() for param in local_model.parameters()])
		grad_list.append(this_grad)
		local_model.zero_grad()
	
	print (len(grad_list))
	end_time = time.time()
	print (f"one by one implementation takes {end_time-start_time}") ## 0.1869s
	# return grad_list
	'''
	### parallel implementation based on opacus
	start_time = time.time()
	data = data.to(device)
	label = label.to(device)
	
	if (args.model_name == 'inception'):
		prediction,_ = local_model(data)
	else:
		prediction = local_model(data)
		
	loss = criterion(prediction, label)
	loss.backward()
	
	grad_sample = []
	for param in local_model.parameters():
		if (param.requires_grad):
			grad_sample.append(param.grad_sample.detach())
	#grad_sample = copy.deepcopy([param.grad_sample.detach() for param in local_model.parameters()])
	
	end_time = time.time()
	local_model.zero_grad() ### ?
	# print (f"batch implementation takes {end_time-start_time}") ## 0.0090s
	### we also need to check if gradient is calculated correctly..
	# sum1 = [this_grad[0] for this_grad in grad_list]
	# sum2 = grad_sample[0]
	# sum1 = torch.stack(sum1)
	# print (sum1.size(),sum2.size())
	# avg1 = torch.mean(sum1,dim=0)
	# avg2 = torch.mean(sum2,dim=0)
	# print (avg1.size(),avg2.size())
	# print (torch.equal(avg1,avg2))
	# print (torch.allclose(avg1,avg2))
	# print (torch.nonzero(avg1-avg2))
	# print ((avg1-avg2)[0,:20])
	# print (torch.sum(avg1-avg2))
	del local_model
	gc.collect()
	
	return grad_sample


def attack_experiment():
	import warnings
	warnings.filterwarnings("ignore")
	np.random.seed(seed=12345)
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	### dataset && membership inference data
	membership_attack_number = args.membership_attack_number
	target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
	                         membership_attack_number=membership_attack_number,
	                         cutout=args.cutout, n_holes=args.n_holes, length=args.length,server_name=args.server_name)
	num_classes = len(np.unique(target_dataset.label))
	
	# print ("Data Check")
	# print (np.amax(target_dataset.train_data[0]),np.amin(target_dataset.train_data[0]))
	#print(f"class distribution:{np.bincount(target_dataset.label)}")
	
	user_list = [User(dataset=args.dataset, model_name=args.target_model, id=i) for i in range(args.user_number)]
	assign_part_dataset(target_dataset, user_list)
	
	if (args.model_name =='skin_special'):
		target_model = skin_special(num_classes=num_classes,test_rank=args.test_rank)
		target_model =  ModuleValidator.fix(target_model)
	elif (args.model_name =='retina_special'):
		target_model = retina_special(test_rank=args.test_rank)
	elif (args.model_name == 'inception'):
		target_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
		target_model.aux_logit = False
		for param in target_model.parameters():
			param.requires_grad = False
		target_model.fc = nn.Linear(2048, num_classes)
		target_model.fc.weight.requires_grad = True
		target_model.fc.bias.requires_grad = True
	elif (args.model_name == 'resnet50'):
		target_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)  ## change back to resnet50.
		for param in target_model.parameters():
			param.requires_grad = False
			#print (param.size())
		target_model.fc = nn.Linear(2048, num_classes)
		#target_model.fc = nn.Linear(512,num_classes)
		target_model.fc.weight.requires_grad = True
		target_model.fc.bias.requires_grad = True
	elif (args.model_name == 'alexnet'):
		#target_model = torch.hub.load('pytorch/vision:v0.10.0','alexnet',pretrained=False)
		#target_model = AlexNet(num_classes=num_classes)
		target_model = alexnet(num_classes=num_classes)
	elif (args.model_name == 'densenet121'):
		target_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
		num_features = target_model.classifier.in_features
		target_model.classifier = nn.Linear(num_features,num_classes)
		target_model =  ModuleValidator.fix(target_model)
	elif (args.model_name == 'mobilenetv3'):
		target_model = torchvision.models.mobilenet_v3_small(num_classes=num_classes)
		target_model =  ModuleValidator.fix(target_model)
	elif (args.model_name == 'resnet18'):
		target_model = ResNet18( num_classes = num_classes)
		target_model = ModuleValidator.fix(target_model)
	#elif (args.dataset == 'cifar10'):
		#target_model = alexnet(num_classes=10)
	#elif (args.dataset == 'cifar100'):
		#target_model = alexnet(num_classes=100)
	elif (args.model_name == 'densenet_cifar'):
		target_model = densenet(num_classes=100)
		target_model = ModuleValidator.fix(target_model)
	else:
		print ("SPECIAL MODEL")
		target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
	
	for name,p in target_model.named_parameters():
		print (name,p.size())
	
	pytorch_total_params = sum(p.numel() for p in target_model.parameters())
	print(f"total params {pytorch_total_params}")
	
	target_model, train_acc, test_acc = train_models(user_list, target_model, learning_rate=args.target_learning_rate,decay=args.target_l2_ratio, epochs=args.target_epochs,target_dataset=target_dataset)
	## blackbox auc
	#get_blackbox_auc(user_list=user_list,target_model=target_model)
	
	
	### doing final attacks evaluation
	name_string_prefix = '/home/lijiacheng/whiteboxmi/new_expdata/'
	if (args.server_name == 'ramos'):
		name_string_prefix = '/homes/li2829/scratch/fed/fed_expdata/'
	name_string_mid_str = get_naming_mid_str()
	print(name_string_mid_str)
	result_file = name_string_prefix + name_string_mid_str + str(args.dataset) + '_' + str(args.model_name) + '_' + str(
		args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(args.best_layer) + '_' + 'result_file.txt'
	f = open(result_file, 'w')
	
	'''
	### diff MI attack
	### for this attack, we have no AUC score to report, only accuracy or F1-score. (all_info should contain m_pred and m_true)
	
	# _,_ = diffmi_attack(user_list,target_model,batch_size=20,output_file=f)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_true_20.npy',m_true)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_pred_20.npy',m_pred)
	# print ("diffmi attack batch_size 20 finished")
	# _,_ = diffmi_attack(user_list,target_model,batch_size=2*args.target_batch_size,output_file=f)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_true_200.npy',m_true)
	# np.save('./new_expdata/diffmi_attack'+get_naming_mid_str()+'_m_pred_200.npy',m_pred)
	# print ("diffmi attack batch_size 200 finished")
	
	### nasr fed attack
	### for this attack, we can report AUC score since the output is a probability.
	#nasr_fed_attack(user_list,target_model)
	# print ("nasr attack finished")
	
	### blackbox attack
	#run_blackbox_attacks(user_list, target_model, num_classes=len(np.unique(target_dataset.train_label)), output_file=f)
	'''
	
	### whitebox attack
	from run_attacks import all_analysis
	
	starting_epoch_dict = {'alexnet':100,'densenet_cifar':150,'purchase':1,'retina':40,'skin':40,'kidney':1,'chest':1,'covid':1,'skin_special':1,'inception':1,'resnet50':1}
	
	## we assume the attacker epoch gap is 1
	epochs = ((np.arange(int(args.target_epochs / args.attacker_epoch_gap) - 1) + 1) * args.attacker_epoch_gap)
	#epochs = np.arange(starting_epoch_dict[args.model_name],args.target_epochs)
	print (epochs)
	#epochs = np.array([i for i in range(30,100)])
	#epochs = np.array([100,150,200,250,300])-1
	#epochs = np.array([60,70,80,90,100])-1
	#epochs = np.array([110,120,130,140,150])-1
	#epochs = np.array([300,350,400,450,500])-1
	
	num_layers_dict = {'alexnet': 6, 'densenet_cifar': 100, 'purchase': 4, 'texas': 4, 'fashion_mnist': 4,
	                   'retina': 6,'mnist':4,'mnist_special':2,'medical_mnist':4,'chest':6,'onelayer_cifar':1,
	                   'lenet_cifar':4,'tb':6,'skin':6,'kidney':6,'skin_special':1,'retina_special':1,'covid':6,'resnet50':1,'inception':1,'resnet18':22}
	num_layers = num_layers_dict[args.model_name]
	
	#if (args.whole_nn):
	#	num_layers = 1
	
	if (args.model_name == 'densenet_cifar' and args.active_attacker==0):
		# avg_auc, avg_tpr, all_user_tpr_list =
		all_analysis_layerwise(epochs, name_string_prefix, name_string_mid_str, args.dataset, args.model_name,
		                                                   int(args.target_data_size), int(args.eval_data_size), f=f, num_layers=num_layers,
		                                                   num_user=args.user_number,
		                                                   client_adversary=args.client_adversary, best_layer=args.best_layer,  # this should be args.best_layer
		                                                   comm_round_list=np.arange(
			                                                   int(args.target_data_size / (args.num_step * args.target_batch_size))), active_adversary=args.active_attacker)
	else:
		# avg_auc,avg_tpr,all_user_tpr_list =
		all_analysis(epochs, name_string_prefix, name_string_mid_str, args.dataset, args.model_name,
	                          int(args.target_data_size), int(args.eval_data_size), f=f, num_layers=num_layers,
	                          num_user=args.user_number,
	                          client_adversary=args.client_adversary, best_layer= args.best_layer, # this should be args.best_layer
	                          comm_round_list=np.arange(
		                          int(args.target_data_size / (args.num_step * args.target_batch_size))),active_adversary=args.active_attacker,validation_set_size=args.validation_set_size)
	
	f.close()
	f = open(result_file, 'r')
	for lines in f.readlines():
		print(lines)
	f.close()
	
	#tpr_list_name = get_naming_mid_str() + str(args.dataset) + '_' + str(args.model_name) + '.npy'
	#print (tpr_list_name)
	#np.save(tpr_list_name,all_user_tpr_list)
	#return avg_auc,avg_tpr
	
	#return 0,0

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--server_name',type=str,default='')
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
	parser.add_argument('--validation_set_size',type=int,default=1000)
	
	parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120])
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--pretrained', type=int, default=0)
	parser.add_argument('--temperature_scaling', type=int, default=0)
	parser.add_argument('--early_stopping', type=int, default=0)
	parser.add_argument('--alpha', type=float, default='1.0')
	parser.add_argument('--mixup', type=int, default=0)
	parser.add_argument('--label_smoothing', type=float, default=0)
	# parser.add_argument('--dropout', type=int, default=0)
	parser.add_argument('--cutout', type=int, default=0)
	parser.add_argument('--n_holes', type=int, default=1)
	parser.add_argument('--length', type=int, default=16)
	
	parser.add_argument('--num_step', type=int, default=20)
	
	parser.add_argument('--whitebox', type=int, default=0)
	parser.add_argument('--middle_output', type=int, default=0)
	parser.add_argument('--middle_gradient', type=int, default=0)
	parser.add_argument('--save_exp_data', type=int, default=1)
	parser.add_argument('--test', type=int, default=0)
	
	### data aug params
	parser.add_argument('--data_aug',type=int,default=0)
	
	### fed params
	parser.add_argument('--local_epochs', type=int, default=1)
	parser.add_argument('--user_number', type=int, default=2)
	parser.add_argument('--client_adversary', type=int, default=0)
	parser.add_argument('--noniid', type=int, default=0)
	## if adversary_client is 1, then one client is the adversary, otherwise the server is the adversary
	
	### dpsgd params
	parser.add_argument('--dpsgd', type=int, default=0)
	parser.add_argument('--grad_norm', type=float, default=0) #1e10
	parser.add_argument('--noise_scale', type=float, default=0) #1e-7
	
	### MMD params
	parser.add_argument('--mmd', type=int, default=0)
	parser.add_argument('--mmd_loss_lambda', type=float, default=0)
	
	### signSGD params
	parser.add_argument('--signsgd',type=int,default=0)
	
	### attacker params
	parser.add_argument('--active_attacker', type=int, default=0)
	parser.add_argument('--active_attacker_mislabel', type=int, default=0)
	parser.add_argument('--eval_data_size', type=int, default=100)
	parser.add_argument('--aux_data_size', type=int, default=200)
	parser.add_argument('--active_attacker_epoch', type=int, default=1)
	parser.add_argument('--attacker_epoch_gap', type=int, default=10)
	parser.add_argument('--active_attacker_lr_multiplier', type=float, default=0)
	parser.add_argument('--mislabeling_target_label', type=int, default=-1)
	parser.add_argument('--pre_gd_epochs', type=int, default=10)
	# if mislabeling_target_label == -1, random label
	parser.add_argument('--mislabeling_target_class', type=int, default=-1)
	parser.add_argument('--best_layer', type=int, default=-1)
	parser.add_argument('--unequal', type=int, default=0)
	parser.add_argument('--max_instance_per_batch', type=int, default=100)  # 5 for alexnet.. we need to try resnet18.
	parser.add_argument('--active_attack_param_search', type=int, default=0)
	parser.add_argument('--whole_nn',type=int,default=0)
	
	parser.add_argument('--track_loss', type=int, default=0)
	parser.add_argument('--random_seed', type=int, default=12345)
	parser.add_argument('--repartition',type=int,default=0)
	parser.add_argument('--test_rank',type=int,default=0)
	parser.add_argument('--num_kernels',type=int,default=16)
	args = parser.parse_args()
	print(vars(args))
	
	
	if (args.dataset == 'texas'):
		args.max_instance_per_batch = 10
	if (args.model_name == 'densenet_cifar'):
		args.max_instance_per_batch = 200 ### maybe 10. not sure.
	args.max_instance_per_batch = min(args.max_instance_per_batch, args.eval_data_size)
	
	#torch.set_default_dtype(torch.float64)
		
	### if we pay attention to validation loss and use early stopping,
	### for single user case, we should stop at 20 epochs for purchase and texas /  50 epochs for cifar100 / 50 epochs for cifar10
	### we set the attack_epoch_gap to be 1 for single user case and see how isolation can impact
	### for 10 user case, we should stop at 100 epochs for purchase / ? epochs for  texas / ? epochs for cifar100 / ? epochs for cifar10
	
	
	#random_seed_list = [0,1,2,3,4,5]
	random_seed_list = [args.random_seed]
	#random_seed_list = [1,2,3]
	
	avg_tpr_list = []
	avg_auc_list = []
	
	for this_seed in random_seed_list:
		import torch
		
		torch.manual_seed(this_seed)
		import numpy as np
		
		np.random.seed(this_seed)
		import sklearn
		sklearn.utils.check_random_state(this_seed)
		
		attack_experiment()
		
		#for epoch_num in range(2,30):
		#	args.target_epochs = epoch_num*10
		#	print (f"epochs {args.target_epochs}")
		#avg_auc,avg_tpr = attack_experiment()
		#avg_auc_list.append(avg_auc)
		#avg_tpr_list.append(avg_tpr)
	
	#print(vars(args))
	
	#avg_tpr_list = np.array(avg_tpr_list)
	#avg_auc_list = np.array(avg_auc_list)
	#print (f"avg tpr:{np.average(avg_tpr_list,axis=0)}")
	#print (f"avg auc:{np.average(avg_auc_list,axis=0)}")
