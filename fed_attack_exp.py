import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from opacus import PrivacyEngine
from multi_party_attack import *
from run_attacks import *
from signum import Signum,SignedAdam
import torch.hub
from opacus.validators import ModuleValidator
from run_attacks import all_analysis
from user import *
from model_utils import *

def get_naming_mid_str():
	name_string_mid_str =  str(args.random_seed) + '_' + str(args.noniid) + '_' + ('client_' if (args.client_adversary) else 'server_') + \
	                      (str(args.active_attacker)) + '_' + (str(args.active_attacker_lr_multiplier)) + '_' + \
	                      str(args.user_number) + '_' + str(args.num_step) + '_' + str(args.dpsgd) + '_' + str(
		args.noise_scale) + '_' + str(args.grad_norm) + '_' + str(args.mmd) + '_' + str(
		args.mmd_loss_lambda) + '_' + str(args.mixup) + '_'
	
	if(args.signsgd):
		name_string_mid_str = 'sign_' + name_string_mid_str
	
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
		 # center crop + horizontal clip
		transform_train = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(32),
			transforms.RandomHorizontalFlip(),
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
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))  # the # of data left for generating new split of training data
	assigned_index = []

	for i in range(num_users):
		training_set_size = args.target_data_size
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
		
		this_user.class_weight = np.ones((len(np.unique(dataset.train_label)))) * training_set_size / (
				len(np.unique(dataset.train_label)) * (np.bincount(this_user.train_label) + 1))
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
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
		
def run_blackbox_attacks(user_list, target_model, num_classes, output_file):
	acc = 0
	for user_idx in range(len(user_list)):
		black_ref = blackbox_attack(args.eval_data_size, 'global_prob', num_classes=num_classes)
		total_confidences, total_classes, total_labels = black_ref.get_attack_input(target_model, user_list[user_idx])
		acc += black_ref.attack(total_confidences=total_confidences, total_classes=total_classes,
		                        total_labels=total_labels, output_file=output_file)  ### labels here is the true label
	return acc / len(user_list)

def run_multi_party_attacks(user_list, target_model, epoch, user_update_list, user_model_list, ori_model_weight_dict,
                            server_attacker=False, attack_loader_list=[], comm_round_idx=0,best_layer=None,save_path_prefix='./expdata/'):
	naming_str = get_naming_mid_str() + str(comm_round_idx) + '_' + str(epoch + 1) + '_' + str(
		args.dataset) + '_' + str(args.target_data_size) + '_' + str(args.eval_data_size) + '_' + str(
		args.model_name) + '.npy'
	
	
	### for transfer learning, we only look at the last layer, there is a 2 because it has weights and biases
	if (args.model_name == 'inception' or args.model_name == 'resnet50'):
		user_update_list = [user_update_list[i][-2:]  for i in range(len(user_update_list))]

	if (server_attacker):
		all_info, all_label = multi_party_member_attack(user_list, target_model, batch_size=args.target_batch_size,
		                                                user_update_list=user_update_list,
		                                                get_gradient_func=get_gradient,
		                                                attack_loader_list=attack_loader_list, user_total_instance=args.num_step * args.target_batch_size,
		                                                max_instance_per_batch=args.max_instance_per_batch,best_layer=best_layer,test_rank=args.test_rank,
		                                                whole_nn=args.whole_nn)
		np.save(save_path_prefix + 'all_info_multi_party_member_attack_' + naming_str, all_info)
		np.save(save_path_prefix + 'all_label_multi_party_member_attack_' + naming_str, all_label)
		
		loss_info, loss_label = multi_party_member_loss_attack(user_list, target_model,
		                                                       batch_size=args.target_batch_size,
		                                                       get_gradient_func=get_gradient,
		                                                       user_model_list=user_model_list,
		                                                       attack_loader_list=attack_loader_list,
		                                                       max_instance_per_batch=args.max_instance_per_batch,
		                                                       model_name=args.model_name)
		np.save(save_path_prefix + 'loss_info_multi_party_member_attack_' + naming_str, loss_info)
		np.save(save_path_prefix + 'loss_label_multi_party_member_attack_' + naming_str, loss_label)
		
		
		
		### this is to use validation dataset to perform the attack, so we use validation dataset to estimate the distribution
		### this part can be skipped
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
		np.save(save_path_prefix + 'all_info_multi_party_member_attack_' + naming_str, all_info)
		np.save(save_path_prefix + 'all_label_multi_party_member_attack_' + naming_str, all_label)
		
		loss_info, loss_label = multi_party_member_loss_attack(user_list[:-1], target_model,
		                                                       batch_size=args.target_batch_size,
		                                                       user_update_list=[sum_user_update_list],
		                                                       get_gradient_func=get_gradient,
		                                                       user_model_list=[temp_sum_weights],
		                                                       attack_loader_list=attack_loader_list,
		                                                       max_instance_per_batch=args.max_instance_per_batch,
		                                                       model_name=args.model_name)
		np.save(save_path_prefix +'loss_info_multi_party_member_attack_' + naming_str, loss_info)
		np.save(save_path_prefix + 'loss_label_multi_party_member_attack_' + naming_str, loss_label)
		
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
		noisy_pred = F.softmax(model(noisy_img),dim=1)
		noisy_probs =  np.array([noisy_pred[i,this_label].detach().item() for i in range(repeat_times)])
		### larger loss means smaller probs, so we count the number of times for smaller probs.
		this_count = len(np.arange(len(noisy_probs))[noisy_probs<probs[img_idx]])
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
			probs = np.array([preds[i,labels[i]].detach().item() for i in range(len(labels))])
			member_probs.append(probs)
	
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images),dim=1)
			probs = np.array([preds[i,labels[i]].detach().item() for i in range(len(labels))])
			nonmember_probs.append(probs)
		
		member_probs = np.concatenate(member_probs).flatten()
		nonmember_probs = np.concatenate(nonmember_probs).flatten()
		min_len = min(len(member_probs),len(nonmember_probs))
		min_len = min(min_len,args.eval_data_size)
		member_index = np.random.choice(len(member_probs),min_len,replace=False)
		nonmember_index = np.random.choice(len(nonmember_probs),min_len,replace=False)
		probs = np.concatenate((member_probs[member_index],nonmember_probs[nonmember_index]),axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)),np.zeros((min_len))),axis=0).astype(np.int64).flatten()
		from sklearn.metrics import roc_auc_score, roc_curve
		auc_score = roc_auc_score(labels,probs)
		fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
		return_tpr = get_tpr(pred=probs,label=labels)
		
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
		min_len = min(len(member_counts), len(nonmember_counts))
		min_len = min(min_len, args.eval_data_size)
		member_index = np.random.choice(len(member_counts), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_counts), min_len, replace=False)
		counts = np.concatenate((member_counts[member_index], nonmember_counts[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
	
		auc_score = roc_auc_score(labels, counts)
		fpr, tpr, thresholds = roc_curve(labels, counts, pos_label=1)
		return_tpr = get_tpr(pred=counts, label=labels)
		merlin_auc.append(auc_score)
		merlin_tpr.append(return_tpr)

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
			probs = modified_entropy(preds,labels)
			member_probs.append(probs)
		
		for (images, labels, _) in user_list[idx].test_data_loader:
			images = images.cuda()
			labels = labels.cuda()
			preds = F.softmax(target_model(images), dim=1)
			probs = modified_entropy(preds,labels)
			nonmember_probs.append(probs)
		
		member_probs = np.concatenate(member_probs).flatten()
		nonmember_probs = np.concatenate(nonmember_probs).flatten()
		min_len = min(len(member_probs), len(nonmember_probs))
		min_len = min(min_len, args.eval_data_size)
		member_index = np.random.choice(len(member_probs), min_len, replace=False)
		nonmember_index = np.random.choice(len(nonmember_probs), min_len, replace=False)
		probs = np.concatenate((member_probs[member_index], nonmember_probs[nonmember_index]), axis=0).flatten()
		labels = np.concatenate((np.ones((min_len)), np.zeros((min_len))), axis=0).astype(np.int64).flatten()
	
		auc_score = roc_auc_score(labels, probs)
		fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
		return_tpr = get_tpr(pred=probs, label=labels)
		
		song_auc.append(auc_score)
		song_tpr.append(return_tpr)
	
	print(f"modified entropy attack: avg auc {np.average(np.array(song_auc))}, avg tpr {np.average(np.array(song_tpr))} at fpr {10 / min_len}")
	print(f"auc std : {np.std(np.array(song_auc))}, tpr std :{np.std(np.array(song_tpr))}")


def train_models(user_list, target_model, learning_rate, decay, epochs, class_weights=None,target_dataset=None):
	num_users = len(user_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.to(device)
	momentum = 0.9
	
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
				noise_multiplier=args.noise_scale,
				max_grad_norm=args.grad_norm)
	
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
				noise_multiplier=args.noise_scale,
				max_grad_norm=args.grad_norm)
		
	### start training
	for epoch in tqdm(range(epochs)):
		if (args.repartition):
			repartition_dataset(user_list)
		
		ori_target_model_state_dict = target_model.state_dict()  ### should we copy this?
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
					ban_list = update_weights(current_model_weights=ori_target_model_state_dict,
					                             model=user_list[idx].model,
					                             optimizer=user_list[idx].optim, train_loader=train_data_loader,
					                             local_epochs=args.local_epochs, mixup=args.mixup,
					                             num_step=args.num_step,
					                             class_weights=user_list[idx].class_weight, unequal=args.unequal,
					                             model_name=args.model_name) ### model name is specifically for inception..
					user_list[idx].update_ban_list(ban_list)
					torch.cuda.empty_cache()
					
				local_weights.append((user_list[idx].model.state_dict()))
			
		
			# epoch level attack
			if (epoch % args.attacker_epoch_gap == 0 and comm_round_idx + 1 == comm_round_per_epoch and epoch > 0):  # and epoch > 0 NEED TO RESTORE
				### this atack requires # of steps = # of batches
				num_batches = int(args.target_data_size / args.target_batch_size)
				if (args.num_step != num_batches):
					print("epoch attack: num steps not equal to num batches")
					exit(0)
				
				user_update_list = get_user_update_list(ori_target_model_state_dict, local_weights, learning_rate=learning_rate, num_batches=num_batches)
				
				local_model = copy.deepcopy(target_model)
				run_multi_party_attacks(user_list, local_model, epoch, user_update_list,
				                        user_model_list=local_weights,
				                        server_attacker=1 - args.client_adversary,
				                        ori_model_weight_dict=ori_target_model_state_dict,
				                        attack_loader_list=[],
				                        comm_round_idx=comm_round_idx,best_layer=args.best_layer)
				
			### update global weights
			global_weights = average_weights(local_weights)
			target_model.load_state_dict(global_weights)
			
	train_acc, test_acc = get_train_test_acc(user_list, target_model)
	print(f"train acc {train_acc},test acc {test_acc}")
	
	return target_model, train_acc, test_acc


def get_gradient(target_model, data, label):
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
	### parallel implementation based on opacus
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
	local_model.zero_grad()
	return grad_sample


def attack_experiment():
	import warnings
	warnings.filterwarnings("ignore")
	np.random.seed(seed=12345)
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	### dataset && membership inference data
	membership_attack_number = args.membership_attack_number
	target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
	                         membership_attack_number=0,path='/home/lijiacheng/dataset/')
	num_classes = len(np.unique(target_dataset.label))
	
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
		target_model.fc = nn.Linear(2048, num_classes)
		target_model.fc.weight.requires_grad = True
		target_model.fc.bias.requires_grad = True
	elif (args.model_name == 'alexnet'):
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
	elif (args.model_name == 'densenet_cifar'):
		target_model = densenet(num_classes=100)
		target_model = ModuleValidator.fix(target_model)
	else:
		target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
	
	### train model
	target_model, train_acc, test_acc = train_models(user_list, target_model, learning_rate=args.target_learning_rate,decay=args.target_l2_ratio, epochs=args.target_epochs,target_dataset=target_dataset)

	### doing final attacks evaluation
	name_string_prefix = './expdata/' ### this is the path to save experimental data
	name_string_mid_str = get_naming_mid_str()
	
	epochs = ((np.arange(int(args.target_epochs / args.attacker_epoch_gap) - 1) + 1) * args.attacker_epoch_gap)
	num_layers_dict = {'alexnet': 6, 'densenet_cifar': 100, 'purchase': 4, 'texas': 4, 'fashion_mnist': 4,
	                   'retina': 6,'mnist':4,'mnist_special':2,'medical_mnist':4,'chest':6,'onelayer_cifar':1,
	                   'lenet_cifar':4,'tb':6,'skin':6,'kidney':6,'skin_special':1,'retina_special':1,'covid':6,'resnet50':1,'inception':1,'resnet18':22}
	num_layers = num_layers_dict[args.model_name]
	
	if (args.model_name == 'densenet_cifar' and args.active_attacker==0):
		
		all_analysis_layerwise(epochs, name_string_prefix, name_string_mid_str, args.dataset, args.model_name,
		                                                   int(args.target_data_size), int(args.eval_data_size), num_layers=num_layers,
		                                                   num_user=args.user_number,
		                                                   client_adversary=args.client_adversary, best_layer=args.best_layer,  # this should be args.best_layer
		                                                   comm_round_list=np.arange(
			                                                   int(args.target_data_size / (args.num_step * args.target_batch_size))), active_adversary=args.active_attacker)
	else:
		all_analysis(epochs, name_string_prefix, name_string_mid_str, args.dataset, args.model_name,
	                          int(args.target_data_size), int(args.eval_data_size), num_layers=num_layers,
	                          num_user=args.user_number,
	                          client_adversary=args.client_adversary, best_layer= args.best_layer, # this should be args.best_layer
	                          comm_round_list=np.arange(
		                          int(args.target_data_size / (args.num_step * args.target_batch_size))),active_adversary=args.active_attacker,validation_set_size=args.validation_set_size)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--server_name',type=str,default='')
	parser.add_argument('--target_data_size', type=int, default=3000)
	parser.add_argument('--target_model', type=str, default='cnn')
	parser.add_argument('--target_learning_rate', type=float, default=0.01)
	parser.add_argument('--target_batch_size', type=int, default=100)
	parser.add_argument('--target_epochs', type=int, default=20)
	parser.add_argument('--target_l2_ratio', type=float, default=5e-4)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--validation_set_size',type=int,default=1000)
	
	parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120])
	parser.add_argument('--model_name', type=str, default='alexnet')
	parser.add_argument('--pretrained', type=int, default=0)
	parser.add_argument('--alpha', type=float, default='1.0')
	parser.add_argument('--mixup', type=int, default=0)
	parser.add_argument('--label_smoothing', type=float, default=0)
	parser.add_argument('--cutout', type=int, default=0)
	parser.add_argument('--n_holes', type=int, default=1)
	parser.add_argument('--length', type=int, default=16)
	parser.add_argument('--num_step', type=int, default=20)
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
	parser.add_argument('--max_instance_per_batch', type=int, default=100)
	parser.add_argument('--eval_data_size', type=int, default=100)
	parser.add_argument('--attacker_epoch_gap', type=int, default=10)
	
	parser.add_argument('--random_seed', type=int, default=12345)
	
	###ablation study params
	parser.add_argument('--repartition',type=int,default=0)
	parser.add_argument('--track_loss', type=int, default=0)
	parser.add_argument('--test_rank',type=int,default=0)
	parser.add_argument('--num_kernels',type=int,default=16)
	
	#useless params
	parser.add_argument('--attack_learning_rate', type=float, default=0.001)
	parser.add_argument('--membership_attack_number', type=int, default=0)
	parser.add_argument('--active_attacker', type=int, default=0)
	parser.add_argument('--active_attacker_mislabel', type=int, default=0)
	parser.add_argument('--aux_data_size', type=int, default=200)
	parser.add_argument('--active_attacker_epoch', type=int, default=1)
	parser.add_argument('--active_attacker_lr_multiplier', type=float, default=0)
	parser.add_argument('--mislabeling_target_label', type=int, default=-1)
	parser.add_argument('--pre_gd_epochs', type=int, default=10)
	parser.add_argument('--mislabeling_target_class', type=int, default=-1)
	parser.add_argument('--best_layer', type=int, default=-1)
	parser.add_argument('--unequal', type=int, default=0)
	parser.add_argument('--active_attack_param_search', type=int, default=0)
	parser.add_argument('--whole_nn',type=int,default=0)
	
	
	
	args = parser.parse_args()
	print(vars(args))
	
	### this is to set the batch size when calculating the gradient separately
	if (args.dataset == 'texas'):
		args.max_instance_per_batch = 10
	if (args.model_name == 'densenet_cifar'):
		args.max_instance_per_batch = 200
	args.max_instance_per_batch = min(args.max_instance_per_batch, args.eval_data_size)
	
	
	random_seed_list = [args.random_seed]
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
		