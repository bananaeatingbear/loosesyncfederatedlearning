import argparse
from data import dataset
from model import *
from utils import *
from user import *
from data import *
from model_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import copy
import pytorch_lightning as pl
from scipy.stats import entropy

def get_naming_mid_str():
	name_string_mid_str = str(args.user_number) + '_' + str(args.num_step) + '_'
	
	if (args.local_epochs>1):
		name_string_mid_str = name_string_mid_str + str(args.local_epochs) + '_'
		
	if (args.model_sampling=='simplex'):
		name_string_mid_str = name_string_mid_str  + 'simplex_sampling_'
		
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
			'celeb' in dataset.dataset_name) or dataset.dataset_name == 'sat6' or dataset.dataset_name =='retina' or  (
			'fashion_product' in dataset.dataset_name) or dataset.dataset_name == 'intel' or dataset.dataset_name == 'gtsrb'):
		transform_train = transforms.ToTensor()
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
	
	if (dataset.dataset_name == 'purchase' or dataset.dataset_name == 'texas'):
		transform_train = None
		transform_test = None
		target_transform = None
		print ("1D dataset!")
	
	num_users = len(user_list)
	
	training_set_size = args.target_data_size
	index_left = np.arange(len(dataset.train_label))
	assigned_index = []
	
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
		
		this_user.class_weight = np.ones((len(np.unique(this_user.train_label)))) * training_set_size / (len(np.unique(this_user.train_label)) * np.bincount(this_user.train_label))
		
		this_user.test_data = dataset.test_data
		this_user.test_label = dataset.test_label
		assigned_index.append(this_user_train_index)
		
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
		this_user.train_eval_data_loader = torch.utils.data.DataLoader(train_eval, batch_size=args.target_batch_size, shuffle=False,
		                                                               num_workers=1)
		this_user.train_loader_in_order = torch.utils.data.DataLoader(train, batch_size=args.target_batch_size,
		                                                              shuffle=False, num_workers=1)
		
	### check remaining unassigned data
	dataset.remaining_index = index_left
	valid_data = dataset.train_data[index_left]
	valid_label = dataset.train_label[index_left]
	
	for i in range(len(user_list)):
		user_list[i].valid_data = valid_data
		user_list[i].valid_label = valid_label
		user_list[i].validation_dataset = part_pytorch_dataset(valid_data, valid_label, train=False,
		                                  transform=transform_test, target_transform=target_transform)
		user_list[i].validation_data_loader =  torch.utils.data.DataLoader(user_list[i].validation_dataset, batch_size=args.target_batch_size,
		                                                              shuffle=False, num_workers=1)
		
	### prepare data for csgmcmc
	assigned_index = np.array(assigned_index).flatten()
	np.save(f'./csgmcmc/{dataset.dataset_name}_{training_set_size}_train_data.npy',dataset.train_data[assigned_index])
	np.save(f'./csgmcmc/{dataset.dataset_name}_{training_set_size}_train_label.npy',dataset.train_label[assigned_index])
	np.save(f'./csgmcmc/{dataset.dataset_name}_{training_set_size}_test_data.npy',dataset.test_data)
	np.save(f'./csgmcmc/{dataset.dataset_name}_{training_set_size}_test_label.npy',dataset.test_label)
	np.save(f'./csgmcmc/{dataset.dataset_name}_{training_set_size}_valid_data.npy',valid_data)
	np.save(f'./csgmcmc/{dataset.dataset_name}_{training_set_size}_valid_label.npy',valid_label)

def get_prediction(model,user_list,train_data=True):
	pred = []
	true_labels = []
	for this_user in user_list:
		if (train_data):
			data_loader = this_user.train_eval_data_loader
		else:
			data_loader = this_user.test_data_loader
		for images,labels,_ in data_loader:
			images = images.cuda()
			labels = labels.cuda()
			this_pred = F.softmax(model(images),dim=1).detach()
			pred.append(this_pred)
			true_labels.append(labels)
	
	pred = torch.vstack(pred)
	true_labels = torch.flatten(torch.cat(true_labels))
	#print (pred.shape,true_labels.shape)
	return pred,true_labels

def get_accuracy(truth, pred):
	assert len(truth)==len(pred)
	right = 0
	for i in range(len(truth)):
		if truth[i]==pred[i]:
			right += 1.0
	return right/len(truth)

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

def get_uncertainty_prediction(model,loader):
	pred = []
	true_labels = []
	for images,labels,_ in loader:
		images = images.cuda()
		labels = labels.cuda()
		this_pred = F.softmax(model(images),dim=1).detach()
		pred.append(this_pred)
		true_labels.append(labels)
	
	pred = torch.vstack(pred)
	true_labels = torch.flatten(torch.cat(true_labels))
	#print (pred.shape,true_labels.shape)
	return pred,true_labels
	
def get_uncertainty_loader(dataset):
	if (dataset == 'cifar10'):
		num_classes = 10
		test_data = np.load('/home/lijiacheng/dataset/gtsrb_test_images.npy').astype(np.uint8)
		test_label = np.load('/home/lijiacheng/dataset/gtsrb_test_labels.npy').astype(np.int64)
		transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		target_transform = transforms.ToTensor()
		sampled_class = np.sort(np.random.choice(np.unique(test_label), num_classes, replace=False))
		sampled_test_index = []
		sampled_test_label = []
		for idx, this_class in enumerate(sampled_class):
			this_class_index = np.arange(len(test_label))[test_label == this_class]
			sampled_test_index.append(this_class_index)
			sampled_test_label.append(np.ones(len(this_class_index)) * idx)
		sampled_test_index = np.hstack(sampled_test_index)
		sampled_test_data = test_data[sampled_test_index, :, :, :]
		sampled_test_label = np.hstack(sampled_test_label)
		gtsrb_testset = part_pytorch_dataset(sampled_test_data, sampled_test_label, train=False, transform=transform_test,
		                                     target_transform=target_transform)
		gtsrb_testloader = torch.utils.data.DataLoader(gtsrb_testset, batch_size=100, shuffle=False, num_workers=0)
		loader = gtsrb_testloader
	if (dataset == 'intel'):
		num_classes = 6
		test_data = np.load('/home/lijiacheng/dataset/gtsrb_test_images.npy').astype(np.uint8)
		test_label = np.load('/home/lijiacheng/dataset/gtsrb_test_labels.npy').astype(np.int64)
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
		sampled_class = np.sort(np.random.choice(np.unique(test_label), num_classes, replace=False))
		sampled_test_index = []
		sampled_test_label = []
		for idx, this_class in enumerate(sampled_class):
			this_class_index = np.arange(len(test_label))[test_label == this_class]
			sampled_test_index.append(this_class_index)
			sampled_test_label.append(np.ones(len(this_class_index)) * idx)
		sampled_test_index = np.hstack(sampled_test_index)
		sampled_test_data = test_data[sampled_test_index, :, :, :]
		sampled_test_label = np.hstack(sampled_test_label)
		gtsrb_testset = part_pytorch_dataset(sampled_test_data, sampled_test_label, train=False, transform=transform_test,
		                                     target_transform=target_transform)
		gtsrb_testloader = torch.utils.data.DataLoader(gtsrb_testset, batch_size=100, shuffle=False, num_workers=0)
		loader = gtsrb_testloader
	if (dataset == 'sat6'):
		num_classes = 6
		test_data = np.load('/home/lijiacheng/dataset/mnist_test_data.npy').astype(np.uint8)
		test_label = np.load('/home/lijiacheng/dataset/mnist_test_label.npy').astype(np.int64)
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
		sampled_class = np.sort(np.random.choice(np.unique(test_label), num_classes, replace=False))
		sampled_test_index = []
		sampled_test_label = []
		for idx, this_class in enumerate(sampled_class):
			this_class_index = np.arange(len(test_label))[test_label == this_class]
			sampled_test_index.append(this_class_index)
			sampled_test_label.append(np.ones(len(this_class_index)) * idx)
		sampled_test_index = np.hstack(sampled_test_index)
		sampled_test_data = test_data[sampled_test_index]
		sampled_test_label = np.hstack(sampled_test_label)
		resized_sampled_test_data = [np.repeat(np.reshape(this_img, (28, 28, 1)), 4, axis=-1) for this_img in sampled_test_data]
		resized_sampled_test_data = np.array(resized_sampled_test_data)
		gtsrb_testset = part_pytorch_dataset(resized_sampled_test_data, sampled_test_label, train=False, transform=transform_test,
		                                     target_transform=target_transform)
		gtsrb_testloader = torch.utils.data.DataLoader(gtsrb_testset, batch_size=100, shuffle=False, num_workers=0)
		loader = gtsrb_testloader
	if (dataset == 'retina'):
		num_classes = 4
		test_data = np.load('/home/lijiacheng/dataset/gtsrb_test_images.npy').astype(np.uint8)
		test_label = np.load('/home/lijiacheng/dataset/gtsrb_test_labels.npy').astype(np.int64)
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
		sampled_class = np.sort(np.random.choice(np.unique(test_label), num_classes, replace=False))
		sampled_test_index = []
		sampled_test_label = []
		for idx, this_class in enumerate(sampled_class):
			this_class_index = np.arange(len(test_label))[test_label == this_class]
			sampled_test_index.append(this_class_index)
			sampled_test_label.append(np.ones(len(this_class_index)) * idx)
		sampled_test_index = np.hstack(sampled_test_index)
		sampled_test_data = test_data[sampled_test_index, :, :, :]
		import cv2
		resized_sampled_test_data = [cv2.resize(this_img, (64, 64)) for this_img in sampled_test_data]
		resized_sampled_test_data = np.array(resized_sampled_test_data)
		sampled_test_label = np.hstack(sampled_test_label)
		gtsrb_testset = part_pytorch_dataset(resized_sampled_test_data, sampled_test_label, train=False, transform=transform_test,
		                                     target_transform=target_transform)
		gtsrb_testloader = torch.utils.data.DataLoader(gtsrb_testset, batch_size=100, shuffle=False, num_workers=0)
		loader = gtsrb_testloader
	if (dataset == 'fashion_mnist'):
		num_classes = 10
		test_data = np.load('/home/lijiacheng/dataset/mnist_test_data.npy').astype(np.uint8)
		test_label = np.load('/home/lijiacheng/dataset/mnist_test_label.npy').astype(np.int64)
		transform_test = transforms.ToTensor()
		target_transform = transforms.ToTensor()
		sampled_class = np.sort(np.random.choice(np.unique(test_label), num_classes, replace=False))
		sampled_test_index = []
		sampled_test_label = []
		for idx, this_class in enumerate(sampled_class):
			this_class_index = np.arange(len(test_label))[test_label == this_class]
			sampled_test_index.append(this_class_index)
			sampled_test_label.append(np.ones(len(this_class_index)) * idx)
		sampled_test_index = np.hstack(sampled_test_index)
		sampled_test_data = test_data[sampled_test_index, :, :]
		sampled_test_label = np.hstack(sampled_test_label)
		gtsrb_testset = part_pytorch_dataset(sampled_test_data, sampled_test_label, train=False, transform=transform_test,
		                                     target_transform=target_transform)
		gtsrb_testloader = torch.utils.data.DataLoader(gtsrb_testset, batch_size=100, shuffle=False, num_workers=0)
		loader = gtsrb_testloader
	return loader


def model_evaluation(models,user_list,exp_name='sgd'):
	print ("----------")
	### train loss
	pred_list = []
	num_model = len(models)
	for i in range(num_model):
		pred, truth_res =get_prediction(models[i],user_list)
		pred_list.append(pred)
	
	avg_pred = sum(pred_list) / num_model
	values, pred_label = torch.max(avg_pred, dim=1)
	pred_res = list(pred_label.data)
	
	acc = get_accuracy(truth_res, pred_res)
	print(f"{exp_name} train acc", acc)
	
	sum_loss = 0
	for i in range(len(avg_pred)):
		this_pred = avg_pred[i]
		this_label = truth_res[i]
		this_loss = -1 * torch.log(this_pred[this_label])
		sum_loss += this_loss.item()
	print(f"{exp_name} avg train loss", sum_loss / len(avg_pred))
	
	### test loss
	pred_list = []
	for i in range(num_model):
		pred, truth_res =get_prediction(models[i],user_list,train_data=False)
		pred_list.append(pred)
	
	avg_pred = sum(pred_list) / num_model
	values, pred_label = torch.max(avg_pred, dim=1)
	pred_res = list(pred_label.data)
	
	acc = get_accuracy(truth_res, pred_res)
	print(f"{exp_name} test acc", acc)
	
	sum_loss = 0
	for i in range(len(avg_pred)):
		this_pred = avg_pred[i]
		this_label = truth_res[i]
		this_loss = -1 * torch.log(this_pred[this_label])
		sum_loss += this_loss.item()
	print(f"{exp_name} avg test loss", sum_loss / len(avg_pred))

	### ECE
	#print (fake.size(),truth_res.size())
	avg_pred = avg_pred.cpu().numpy()
	truth_res = truth_res.cpu().numpy()
	ece = ece_score(avg_pred,truth_res,n_bins=15)
	print (f"{exp_name} ECE score {ece}")
	
	### uncertainty
	uncertainty_loader = get_uncertainty_loader(dataset=args.dataset)
	pred_list = []
	for i in range(num_model):
		pred, truth_res =get_uncertainty_prediction(models[i],uncertainty_loader)
		pred_list.append(pred)

	avg_pred = sum(pred_list) / num_model
	sum_loss = 0
	sum_loss_list = []
	for i in range(len(avg_pred)):
		this_pred = avg_pred[i]
		this_label = truth_res[i]
		this_loss = entropy(this_pred.cpu().numpy())
		sum_loss += this_loss
		sum_loss_list.append(this_loss)
	print(f"{exp_name} avg uncertainty", sum_loss / len(avg_pred))
	#np.save(f'./{exp_name}_uncertainty_cifar10.npy', np.array(sum_loss_list))
	
	### volume test
	if (len(models)>1):
		from model_utils import calculate_volume
		model_weight_list = []
		for i in range(num_model):
			model_weight_list.append(models[i].state_dict())
		zero_prev = {}
		for key,val in model_weight_list[0].items():
			zero_prev[key] = torch.zeros_like(val)
		_,volume = calculate_volume(model_weight_list=model_weight_list,prev_weight_list=[zero_prev],lr=1,max_dim=9)
		print(f"volume test {volume}")
	print ("----------")

def simplex_test(user_list, local_weight, local_model, epoch, num_step, comm_round_idx, test_round=10,
                 ori_model_weight={}):
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
		#bb_acc = run_blackbox_attacks(user_list, local_model, num_classes=len(np.unique(user_list[0].train_label)),
		#                              output_file=None)
		acc.append((train_acc, test_acc, 0))
		loss.append((train_loss, test_loss))
	
	data_name = './simplex/simplex_test_acc_' + naming_str
	acc = np.array(acc)
	np.save(data_name, acc)
	
	data_name = './simplex/simplex_test_loss_' + naming_str
	loss = np.array(loss)
	np.save(data_name, loss)
	# print (acc.shape,loss.shape,naming_str)
	
	### flatness test
	
	if (epoch<args.target_epochs-3):
		return
	
	alpha_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
	acc = []
	loss = []
	
	for user_idx in range(len(user_list)):
		this_acc = []
		this_loss = []
		for this_alpha in alpha_list:
			weight_param = np.array([1-this_alpha, this_alpha])
			weighted_avg_weight = average_weights([ori_model_weight, local_weight[user_idx]], weight_param)
			local_model.load_state_dict(weighted_avg_weight)
			train_acc, test_acc, train_loss, test_loss = get_train_test_acc(user_list, local_model, print_option=False,
			                                                                return_loss=True)
			#bb_acc = run_blackbox_attacks(user_list, local_model, num_classes=len(np.unique(user_list[0].train_label)),
			#                              output_file=None)
			this_acc.append((train_acc, test_acc, 0))
			this_loss.append((train_loss, test_loss))
		acc.append(this_acc)
		loss.append(this_loss)
	
	data_name = './simplex/flatness_test_acc_' + naming_str
	acc = np.array(acc)
	np.save(data_name, acc)
	
	data_name = './simplex/flatness_test_loss_' + naming_str
	loss = np.array(loss)
	np.save(data_name, loss)
	print (acc.shape, loss.shape, naming_str)


def sgmcmc(target_model,user_list):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	momentum = 0
	decay = 5e-4
	learning_rate = args.target_learning_rate
	
	for user_idx in range(len(user_list)):
		user_list[user_idx].model = copy.deepcopy(target_model).to(device)
		this_optim = torch.optim.SGD(user_list[user_idx].model.parameters(), lr=learning_rate, momentum=momentum,
		                             weight_decay=decay)
		user_list[user_idx].optim = this_optim
		for param_group in user_list[user_idx].optim.param_groups:
			param_group['lr'] = learning_rate
	
	### now we start sgmcmc
	
	### csgmcmc start with 0.5 .
	### 0.02377 - 0.01757 (0.046 ~ 0.034)
	### 0.01755 - 0.01224 (0.034 ~ 0.024)
	### 0.01224 - 0.00786 (0.024 ~ 0.014)
	
	# sg_mcmc_lr = np.array([1,0.1,0.09,0.08,0.07,0.06])*args.target_learning_rate
	# sg_mcmc_lr = np.array([1,0.5,0.4,0.3,0.2,0.1])*args.target_learning_rate
	# sg_mcmc_lr = np.array([1,0.9,0.7,0.5,0.3,0.1])*args.target_learning_rate
	# sg_mcmc_lr = np.array([1,0.9,0.7,0.5,0.3,0.1])*0.01
	# sg_mcmc_lr = np.array([1,1,1,1,1,1])*args.target_learning_rate
	# sg_mcmc_lr = np.array([1,0.05,0.04,0.03,0.02,0.01])*args.target_learning_rate
	
	if (args.target_learning_rate == 0.01):
		sg_mcmc_lr = np.array([1, 1, 1, 1, 1, 1]) * args.target_learning_rate
	elif (args.target_learning_rate == 0.1):
		sg_mcmc_lr = np.array([1, 0.9, 0.7, 0.5, 0.3, 0.1]) * args.target_learning_rate
	
	return_models = []
	for sg_step in range(1, 6):
		#print(f"sg step {sg_step}")
		for user_idx in range(len(user_list)):
			### adjust learning rate
			for param_group in user_list[user_idx].optim.param_groups:
				param_group['lr'] = sg_mcmc_lr[sg_step]
			
			### train the model with noise
			train_data_loader = user_list[user_idx].create_new_train_data_loader(batch_size=100)
			w = update_weights_mcmc(current_model_weights=user_list[user_idx].model.state_dict(),
			                        model=user_list[user_idx].model, optimizer=user_list[user_idx].optim,
			                        train_loader=train_data_loader, local_epochs=1,
			                        mixup=args.mixup, num_step=args.num_step,
			                        class_weights=user_list[user_idx].class_weight, lr=sg_mcmc_lr[sg_step], datasize=args.target_data_size, alpha=1)
			user_list[user_idx].model.load_state_dict(w)  ### this is not necessary actually..
			
			if (sg_step == 5):
				return_models.append(copy.deepcopy(user_list[user_idx].model))
			
			model_path = f"./model_checkpoints/{args.model_name}_{args.dataset}_{args.num_step}_{args.user_number}_{args.target_data_size}_{user_idx}_mcmc_{sg_step}.pt"
			torch.save(w, model_path)
			train_acc, test_acc = get_train_test_acc([user_list[user_idx]], user_list[user_idx].model, print_option=False)
			#print(f"SG-MCMC step {sg_step}, user idx {user_idx}, train_acc {train_acc}, test acc {test_acc}")
	
	#print("SGMCMC finished")
	
	return return_models


def train_models(user_list, target_model, learning_rate, decay, epochs,target_dataset=None):
	# print (target_model)
	num_users = len(user_list)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	target_model.type(torch.FloatTensor)
	target_model.to(device)
	
	#if (num_users == 1):
	## enable parallel computation for standard SGD. fed learning in parallel in implemented using flower. This implementation is a sequential fashion of federated learning
	#	target_model = torch.nn.DataParallel(target_model, device_ids=[0, 1, 2]).cuda(0)
	#	device = torch.cuda.current_device()
	
	momentum = 0.9
	
	volume_results = []
	grad_norm_results = []
	cosine_results = []
	loss_list = []
	acc_list = []
	best_model = copy.deepcopy(target_model)
	
	min_loss = 1e20
	max_acc = 0
	
	volume_dim_results = []
	
	### notice that in pytorch, momentum etc. is bound with optimizer, so we need to initialize the optimizer/model for each user
	for user_idx in range(num_users):
		user_list[user_idx].model = copy.deepcopy(target_model)
		this_optim = torch.optim.SGD(user_list[user_idx].model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=decay)
		user_list[user_idx].optim = this_optim
		user_list[user_idx].scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(this_optim, T_max=epochs) # eta_min = 0.1*args.target_learning_rate
		
	
	### start training
	for epoch in tqdm(range(int(epochs/args.local_epochs))):
		
		#if (epoch!=0):
		#	for i in range(len(user_list)):
		#		user_list[i].scheduler.step()
		
		ori_target_model_state_dict = copy.deepcopy(target_model.state_dict())
		## LR schedule
		#if (epoch in args.schedule):
		#	learning_rate = learning_rate / 10
		#	print ("new learning rate = %f" % (learning_rate))
			### set up new learning rate for each user
		#	for user_idx in range(num_users):
		#		for param_group in user_list[user_idx].optim.param_groups:
		#			param_group['lr'] = learning_rate
		
		#print (user_list[0].scheduler.get_last_lr())
		
		#### we need to know the # of steps, # of training set size to perform fed sgd
		
		comm_round_per_epoch = int(args.target_data_size / (args.target_batch_size * args.num_step))
		#comm_round_per_epoch = max(comm_round_per_epoch,1)
		
		for comm_round_idx in range(comm_round_per_epoch):
			local_weights = []
			ori_target_model_state_dict = copy.deepcopy(target_model.state_dict())
			
			### get parameter update from each client
			for idx in range(len(user_list)):
				# create a new dataloader
				train_data_loader = user_list[idx].create_new_train_data_loader(batch_size=args.target_batch_size)
				
				w, ban_list = update_weights(current_model_weights=ori_target_model_state_dict,
				                             model=user_list[idx].model, optimizer=user_list[idx].optim,
				                             train_loader=train_data_loader, local_epochs=args.local_epochs,
				                             mixup=args.mixup, num_step=args.num_step,class_weights=user_list[idx].class_weight)
				
				user_list[idx].update_ban_list(ban_list)
				local_weights.append(copy.deepcopy(w))
			
			## simplex test
			## every one round do simplex test
			#if ((comm_round_idx + 1) == comm_round_per_epoch and epoch > 0):
			#	print (f"epoch {epoch}, comm_round_idx {comm_round_idx}")
			#	simplex_test(user_list, copy.deepcopy(local_weights), copy.deepcopy(target_model), epoch, args.num_step,comm_round_idx, test_round=10, ori_model_weight=ori_target_model_state_dict)
			#	print ("simplex test finished")
			
			## repartition
			#if (args.repartition):
			#	print ("repartition activated")
			#	assign_part_dataset(target_dataset,user_list=user_list)
			#	for idx in range(len(user_list)):
			#		user_list[idx].reset_ban_list()
			
			### calculate volume
			#volume_weights = copy.deepcopy(local_weights)
			#prev_volume_weights =  [copy.deepcopy(ori_target_model_state_dict)]
			#volume_results.append(calculate_volume(volume_weights, prev_volume_weights, lr=learning_rate,num_step=args.num_step))
			
			#volume_dim_results.append(volume_dim_test(volume_weights, prev_volume_weights, lr=learning_rate))
			
			### update model
			if (args.model_sampling == 'avg'):
				global_weights = average_weights(local_weights)
			elif (args.model_sampling == 'simplex'):
				weight_param = simplex_uniform_sampling(len(user_list))
				#print (weight_param)
				global_weights = average_weights(local_weights, weight_param)
			
			target_model.load_state_dict(global_weights)
			
			train_acc, test_acc, valid_acc, train_loss, test_loss, valid_loss = get_train_test_acc(user_list, target_model, return_validation_result=True,return_loss=True,print_option=False)
			loss_list.append((train_loss, test_loss, valid_loss))
			acc_list.append((train_acc, test_acc,valid_acc))
			#print (f"Epoch {epoch}:{train_acc},{test_acc},{valid_acc},{train_loss},{test_loss},{valid_loss}")
		
			if (valid_loss<min_loss):
				#print (f"NEW BEST! epoch {epoch}, old min loss {min_loss}, new min loss {valid_loss}")
				min_loss = valid_loss
				best_model = copy.deepcopy(target_model)
				for idx,this_model_weight in enumerate(local_weights):
					#if (args.local_epochs>1):
					#	model_path = f"./model_checkpoints/{args.model_name}_{args.dataset}_{args.num_step}_{args.user_number}_{args.target_data_size}_{idx}_{args.local_epochs}.pt"
					#else:
					model_path = f"./model_checkpoints/{args.model_name}_{args.dataset}_{args.num_step}_{args.user_number}_{args.target_data_size}_{idx}.pt"
					
					torch.save({'epoch': epoch,'model_state_dict': target_model.state_dict(),'optimizer_state_dict': user_list[idx].optim.state_dict()}, model_path)
		
	#volume_name_str = './simplex/volume_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
	#	args.target_data_size) + '_' + str(
	#	args.model_name) + '.npy'
	#volume_results = np.array(volume_results)
	#np.save(volume_name_str, volume_results)

	#volume_dim_name_str = './simplex/volume_dim_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
	#	args.target_data_size) + '_' + str(
	#	args.model_name) + '.npy'
	#volume_dim_results = np.array(volume_dim_results)
	#np.save(volume_dim_name_str, volume_dim_results)
	
	model_path = f"./model_checkpoints/{args.model_name}_{args.dataset}_{args.num_step}_{args.user_number}_{args.target_data_size}_final.pt"
	torch.save(target_model.state_dict(), model_path)
	
	loss_name_str = './simplex/loss_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	np.save(loss_name_str, np.array(loss_list))
	
	acc_name_str = './simplex/acc_' + get_naming_mid_str() + str(args.dataset) + '_' + str(
		args.target_data_size) + '_' + str(
		args.model_name) + '.npy'
	np.save(acc_name_str, np.array(acc_list))
	
	train_acc, test_acc, train_loss, test_loss, ece_loss = get_train_test_acc(user_list, target_model, return_loss=True,print_option=True,return_ece_loss=True)
	print (f"Epoch {epoch}:{train_acc},{test_acc},{train_loss},{test_loss},{ece_loss}")
	
	return best_model, train_acc, test_acc


def get_gradient(target_model, data, label):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	local_model = copy.deepcopy(target_model)
	local_model = local_model.to(device)
	
	criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
	data = data.to(device)
	label = label.to(device)
	prediction = local_model(data)
	loss = criterion(prediction, label)
	loss.backward()
	this_grad = copy.deepcopy([param.grad for param in local_model.parameters()])
	local_model.zero_grad()
	
	return this_grad


def attack_experiment():
	import warnings
	warnings.filterwarnings("ignore")
	torch.set_printoptions(threshold=5000, edgeitems=20)
	
	### dataset && membership inference data
	membership_attack_number = args.membership_attack_number
	target_dataset = dataset(dataset_name=args.dataset, gpu=args.gpu,
	                         membership_attack_number=membership_attack_number,
	                         cutout=args.cutout, n_holes=args.n_holes, length=args.length)
	
	print (np.bincount(target_dataset.label))
	
	user_list = [User(dataset=args.dataset, model_name=args.target_model, id=i) for i in range(args.user_number)]
	
	assign_part_dataset(target_dataset, user_list)
	
	if (target_dataset.dataset_name == 'cifar100'):
		args.num_classes = 100
		if (args.model_name == 'resnet18'):
			target_model = ResNet18(num_classes=100)
		if (args.model_name == 'resnet20'):
			target_model = resnet(depth=20,num_classes=100)
		if (args.model_name == 'densenet_cifar'):
			target_model = densenet(depth=100,num_classes=100)
		if (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=100)
	elif (target_dataset.dataset_name == 'cifar10'):
		args.num_classes = 10
		if (args.model_name == 'resnet20'):
			target_model = resnet(depth=20,num_classes=10)
		if (args.model_name == 'resnet18'):
			target_model = ResNet18(num_classes=10)
		if (args.model_name == 'densenet_cifar'):
			target_model = densenet(depth=100,num_classes=10)
		if (args.model_name == 'alexnet'):
			target_model = alexnet(num_classes=10)
	else:
		target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))
	
	# print (target_model)
	
	target_model, train_acc, test_acc = train_models(user_list, target_model, learning_rate=args.target_learning_rate,
	                                                 decay=args.target_l2_ratio, epochs=args.target_epochs,target_dataset=target_dataset)
	
	### eval model
	if (args.user_number == 1): ## sgd case
		model_evaluation([target_model],user_list,exp_name='sgd')
		new_target_model = ModelWithTemperature(copy.deepcopy(target_model))
		new_target_model.set_temperature(user_list[0].validation_data_loader)
		model_evaluation([new_target_model], user_list, exp_name='sgd-ts')
	else:
		model_evaluation([target_model],user_list,exp_name='fed-avg')
		### add temperature scaling
		new_target_model = ModelWithTemperature(copy.deepcopy(target_model))
		new_target_model.set_temperature(user_list[0].validation_data_loader)
		### eval model again
		model_evaluation([new_target_model],user_list,exp_name='fed-avg-ts')
		### perform SGMCMC
		model_list = sgmcmc(target_model,user_list)
		### eval models after SGMCMC
		model_evaluation(model_list,user_list,exp_name='fed-sgmcmc')
		### add temperature scaling.. should we add temperature scaling for each model separately?
		new_model_list = [ModelWithTemperature(model_list[i]) for i in range(len(model_list))]
		for i in range(len(new_model_list)):
			new_model_list[i].set_temperature(user_list[0].validation_data_loader)
		### eval model again
		model_evaluation(new_model_list,user_list,exp_name='fed-sgmcmc-ts')
	
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
	parser.add_argument('--eval_data_size',type=int,default=50)
	parser.add_argument('--model_sampling',type=str,default='avg')
	### avg -> directly avg, simplex ->  randomly sample from a simplex, particle -> particle sampling using weight
	parser.add_argument('--repartition',type=int,default=0)
	parser.add_argument('--random_seed', type=int, default=0)
	
	args = parser.parse_args()
	print (vars(args))

	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True
	cudnn.deterministic = True
	import torch
	import numpy as np
	import sklearn
	import random
	#import datetime
	#start_time = datetime.now()
	
	random_seed_list = [0,1,2,3,4]
	
	for this_seed in random_seed_list:
		torch.manual_seed(args.random_seed)
		np.random.seed(args.random_seed)
		sklearn.utils.check_random_state(args.random_seed)
		random.seed(args.random_seed)
		#from datetime import datetime
		attack_experiment()
		
	print (vars(args))

	#end_time = datetime.now()
	#print (start_time,end_time)
	#print (f"time spent {(end_time-start_time).seconds}")