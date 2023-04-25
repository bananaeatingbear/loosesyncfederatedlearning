import numpy as np
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import warnings
from data import *
from scipy.stats import norm
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms
import torch
from model import *
import torch.nn as nn
import copy
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math

def transform_loss_to_prob(x):
	### this can be rewritten as a map function
	### we are using cross entropy loss for all tasks
	new_x = np.exp(-x)
	return new_x


def get_tpr(pred,label):
	# in this function, we assume that higher the pred is, higher chance it is a member.
	sorted_index = np.argsort(pred)[::-1]
	sorted_label = label[sorted_index]
	fpr_count = 10
	# we allow 10 false positive, because 1 false positive will be very unstable.
	tpr_count = 0
	for i in range(len(label)):
		if (sorted_label[i] == 0):
			fpr_count = fpr_count-1
			if (fpr_count == 0):
				break
		else:
			tpr_count+=1
	#print (f" fpr {2*10/len(label)}, tpr {2*tpr_count/len(label)}") ## 2 is because the evaluation set is balanced
	return 2*tpr_count/(len(label))

def report_acc(data, label, non_member_included=False):
	default_runs = 1
	unique_labels = np.unique(label)
	train_index = []
	for this_label in unique_labels:
		this_class_index = np.arange(len(label))[label == this_label]
		this_class_train_index = np.random.choice(this_class_index, int(len(this_class_index) / 2), replace=False)
		train_index.append(this_class_train_index)
	train_index = np.reshape(np.array(train_index), (-1))
	test_index = np.setdiff1d(np.arange(len(label)), train_index)
	
	train_data = data[train_index]
	train_label = label[train_index]
	test_data = data[test_index]
	test_label = label[test_index]
	
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import balanced_accuracy_score
	from sklearn.metrics import f1_score
	
	train_data = np.reshape(train_data, (len(train_label), -1))
	test_data = np.reshape(test_data, (len(test_label), -1))
	# LR
	sum_acc = 0
	for _ in range(default_runs):
		clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced')
		clf.fit(train_data, train_label)
		acc1 = balanced_accuracy_score(test_label, clf.predict(test_data)) * 100
		f1_1 = f1_score(test_label, clf.predict(test_data), average='weighted')
		sum_acc += acc1
	acc1 = sum_acc / default_runs
	
	return acc1

def convert_raw_data_to_rank_based_on_validation(data,valid_data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_valid_raw_data_sorted = np.sort(valid_data[:,i])
		for j in range(data.shape[0]):
			## find data[j,i]'s rank in this_epoch_valid_raw_data_sorted
			this_raw_data = data[j,i]
			for this_rank in range(len(this_epoch_valid_raw_data_sorted)):
				if (this_raw_data<this_epoch_valid_raw_data_sorted[this_rank]):
					break
			if (this_raw_data>this_epoch_valid_raw_data_sorted[-1]):
				this_rank+=1
			rank_results[j,i] = this_rank
	return rank_results

def convert_loss_to_logits(data):
	### this function is the same as the one used in the FIRST PRINCIPLE paper
	logits = np.zeros_like(data)
	
	for i in range(len(data)):
		for j in range(len(data[0])):
			logits[i,j] = np.exp(-data[i,j])
			logits[i,j] = np.log(logits[i,j]/(1-logits[i,j]+1e-5)) ### 1e-5 is for numerical stability
	return data

def convert_raw_data_to_std_based_on_validation(data,valid_data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_valid_data = valid_data[:,i]
		valid_mean = np.average(this_epoch_valid_data)
		valid_std = np.std(this_epoch_valid_data)
		for j in range(data.shape[0]):
			this_raw_data = data[j,i]
			rank_results[j,i] = (this_raw_data-valid_mean)/valid_std
			
	weights = [ np.std(valid_data[:,i]) for i in range(data.shape[1])]
	weights = np.array(weights)
	weight_sum = np.sum(weights)
	weights = np.array([weight_sum for i in range(data.shape[1])])/weights
	
	return rank_results, weights

def normal_cdf(z):
	return 0.5*(1+math.erf(z/sqrt(2)))

def convert_raw_data_to_log_prob_based_on_validation(data,valid_data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_valid_data = valid_data[:,i]
		valid_mean = np.average(this_epoch_valid_data)
		valid_std = np.std(this_epoch_valid_data)
		for j in range(data.shape[0]):
			this_raw_data = data[j,i]
			rank_results[j,i] = np.log10(norm.cdf((this_raw_data-valid_mean)/valid_std))
	weights = np.ones((data.shape[1]))
	return rank_results, weights
	

def convert_raw_data_to_rank(data):
	rank_results = np.zeros_like(data)
	for i in range(data.shape[1]):
		this_epoch_raw_data = data[:,i]
		this_argsort = np.argsort(this_epoch_raw_data)
		cur = 0
		for j in range(len(this_epoch_raw_data)):
			rank_results[this_argsort[j],i] = j
			if (j>=1 and this_epoch_raw_data[this_argsort[j]] == this_epoch_raw_data[this_argsort[j-1]]):
				rank_results[this_argsort[j],i] = cur
			else:
				cur = j
	return rank_results


def read_all_comm_round_data(comm_round_list, prefix, mid_str, feature_str, final_str, have_label=True, label_str=''):
	all_data = []
	all_label = []
	
	if (len(comm_round_list) == 1):
		for comm_round_idx in comm_round_list:
			data_name = prefix + feature_str + mid_str + str(comm_round_idx) + final_str
			data = np.load(data_name, allow_pickle=True)
			all_data.append(data)
			
			label_name = prefix + label_str + mid_str + str(comm_round_idx) + final_str
			label = np.load(label_name)
			all_label.append(label)
		
		all_data = np.array(all_data)
		all_label = np.array(all_label).flatten()
		
		return np.squeeze(all_data), np.squeeze(all_label)


def avg_roc_cal(metric_list):
	num_user = len(metric_list)
	uniform_fpr = [1e-5 * i for i in range(1, 10)] + [1e-4 * i for i in range(1, 10)] + [1e-3 * i for i in range(1, 10)]
	uniform_fpr = uniform_fpr + [1e-2 * i for i in range(1, 100)]
	uniform_tpr = []
	
	for this_fpr in uniform_fpr:
		sum_tpr = 0
		for user_idx in range(num_user):
			this_user_roc_list = np.array(metric_list[user_idx])
			idx = np.argmax(this_user_roc_list[0, :] > this_fpr)
			if (this_user_roc_list[0, idx] > this_fpr):
				idx -= 1
			this_tpr = this_user_roc_list[1, idx]
			sum_tpr += this_tpr
		uniform_tpr.append(sum_tpr / num_user)
	
	uniform_fpr = np.array(uniform_fpr)
	uniform_tpr = np.array(uniform_tpr)
	return uniform_fpr, uniform_tpr

def read_valid_nonmember_data(comm_round_list,epochs,prefix,mid_str,dataset,model,target_data_size,eval_data_size,f,num_layers,num_user,validation_set_size=1000):
	total_data_num = validation_set_size
	num_valid_user = num_user
	all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	
	for epoch_idx, epoch in enumerate(epochs):
		final_str = '_' + str(
			epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
			model) + '.npy'
		
		data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                       feature_str='valid_all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')

		loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='valid_loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')

		data = np.reshape(data, (-1, num_valid_user, num_layers, 7)) ### need to change to 4.
		label = np.reshape(label, (-1))
		loss_data = np.reshape(loss_data, (-1, num_valid_user))
		
		all_epoch_layer_cos[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 0])
		all_epoch_layer_grad_diff[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 1] - data[:, :, :, 2])
		all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
		all_epoch_layer_grad_norm[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 3])
	
	all_epoch_loss = np.nan_to_num(all_epoch_loss)
	all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
	all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
	all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
	
	return all_epoch_layer_cos,all_epoch_layer_grad_diff,all_epoch_layer_grad_norm,all_epoch_loss
	

def all_analysis(epochs, prefix, mid_str, dataset, model, target_data_size, eval_data_size, f=None, special_layers=None,
                 num_layers=12, num_user=5, client_adversary=0, best_layer=0, comm_round_list=[],active_adversary=1,validation_set_size=1000):
	if (client_adversary):
		num_user -= 1
		num_valid_user = 1
	else:
		num_valid_user = num_user
	
	total_data_num = (num_user * 2) * eval_data_size
	all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
	all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs), num_layers))
	for epoch_idx, epoch in enumerate(epochs):
		final_str = '_' + str(
			epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
			model) + '.npy'
		
		data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                       feature_str='all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')

		
		loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')
		
		data = np.reshape(data, (-1, num_valid_user, num_layers, 7))
		label = np.reshape(label, (-1))
		loss_data = np.reshape(loss_data, (-1, num_valid_user))
		loss_label = np.reshape(loss_label, (-1))
		
		all_epoch_layer_cos[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 0])
		all_epoch_layer_grad_diff[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 1] - data[:, :, :, 2])
		all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
		all_epoch_layer_grad_norm[:, :, epoch_idx, :] = copy.deepcopy(data[:, :, :, 3])
		all_epoch_label = label
	
	all_epoch_loss = np.nan_to_num(all_epoch_loss)
	all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
	all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
	all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
	
	if (not active_adversary):
		for best_layer in range(num_layers):
			print (f"current layer {best_layer}")
			#print ("set based rank")
			all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
			                                         label, best_layer, eval_data_size, num_user, f)
			
	else:
		for best_layer in range(num_layers):
			print(f"current layer {best_layer}")
			#print("set based rank")
			all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
			                                         label, best_layer, eval_data_size, num_user, f)
	
	#if (client_adversary):
	#    return all_epoch_analysis_client_adversary(all_epoch_cos, all_epoch_grad_diff, all_epoch_loss,
	#                                                all_epoch_grad_norm, all_epoch_target_loss,
	#                                               all_epoch_target_after_loss, all_epoch_label,
	#                                               eval_data_size=eval_data_size, num_user=num_valid_user, f=f)
	#else:
	#    return all_epoch_analysis_server_adversary(all_epoch_cos, all_epoch_grad_diff, all_epoch_loss,
	#                                               all_epoch_grad_norm, all_epoch_target_loss,
	#                                               all_epoch_target_after_loss, all_epoch_label,
	#                                               eval_data_size=eval_data_size, num_user=num_valid_user, f=f)
	# per_layer_analysis(all_epoch_cos, all_epoch_grad_diff, all_epoch_loss, all_epoch_target_loss, all_epoch_label,
	#                   all_epoch_layer_cos, all_epoch_layer_grad_diff, member_index=num_user * eval_data_size)
	# loss_analysis(all_epoch_cos,all_epoch_grad_diff,all_epoch_loss,all_epoch_target_loss,all_epoch_target_after_loss,all_epoch_label,member_index = num_user*eval_data_size)

def all_analysis_layerwise(epochs, prefix, mid_str, dataset, model, target_data_size, eval_data_size, f=None, special_layers=None,
                 num_layers=12, num_user=5, client_adversary=0, best_layer=0, comm_round_list=[], active_adversary=1):
	if (client_adversary):
		num_user -= 1
		num_valid_user = 1
	else:
		num_valid_user = num_user
	
	if (not active_adversary):
		for best_layer in range(num_layers):
			total_data_num = (num_user * 2) * eval_data_size
			all_epoch_loss = np.zeros((total_data_num, num_valid_user, len(epochs)))
			all_epoch_layer_cos = np.zeros((total_data_num, num_valid_user, len(epochs),1))
			all_epoch_layer_grad_diff = np.zeros((total_data_num, num_valid_user, len(epochs),1))
			all_epoch_layer_grad_norm = np.zeros((total_data_num, num_valid_user, len(epochs),1))
			for epoch_idx, epoch in enumerate(epochs):
				final_str = '_' + str(
					epoch + 1) + '_' + str(dataset) + '_' + str(target_data_size) + '_' + str(eval_data_size) + '_' + str(
					model) + '.npy'
		
				data, label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                       feature_str='all_info_multi_party_member_attack_',
		                                       final_str=final_str, label_str='all_label_multi_party_member_attack_')
		
				loss_data, loss_label = read_all_comm_round_data(comm_round_list, prefix, mid_str,
		                                                 feature_str='loss_info_multi_party_member_attack_',
		                                                 final_str=final_str,
		                                                 label_str='loss_label_multi_party_member_attack_')
		

				data = np.reshape(data, (-1, num_valid_user, num_layers, 7))
				label = np.reshape(label, (-1))
				loss_data = np.reshape(loss_data, (-1, num_valid_user))
				loss_label = np.reshape(loss_label, (-1))
				
				all_epoch_layer_cos[:, :, epoch_idx,0] = copy.deepcopy(data[:, :, best_layer, 0])
				all_epoch_layer_grad_diff[:, :, epoch_idx,0] = copy.deepcopy(data[:, :, best_layer, 1] - data[:, :, best_layer, 2])
				all_epoch_loss[:, :, epoch_idx] = copy.deepcopy(loss_data)
				all_epoch_layer_grad_norm[:, :, epoch_idx,0] = copy.deepcopy(data[:, :, best_layer, 3])
	
			all_epoch_loss = np.nan_to_num(all_epoch_loss)
			all_epoch_layer_grad_norm = np.nan_to_num(all_epoch_layer_grad_norm)
			all_epoch_layer_grad_diff = np.nan_to_num(all_epoch_layer_grad_diff)
			all_epoch_layer_cos = np.nan_to_num(all_epoch_layer_cos)
		
			print(f"current layer {best_layer}")
			#print("set based rank")
			all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
	                                         label, 0, eval_data_size, num_user, f)


def all_epoch_analysis_all_adversary_auc_tpr(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
                                              label, best_layer, eval_data_size, num_user, f):
	auc_list = []
	acc_list = []
	tpr_list = []
	all_user_roc_list = []
	num_instance = all_epoch_layer_cos.shape[0]
	num_user = all_epoch_layer_cos.shape[1]
	num_layer = all_epoch_layer_cos.shape[3]
	num_epochs = all_epoch_layer_cos.shape[2]
	
	all_user_tpr_list = []
	
	for this_user_idx in range(num_user):
		
		if (num_user != 1):
			this_user_member_index = np.arange(num_instance)[this_user_idx * eval_data_size:(this_user_idx + 1) * eval_data_size]
			this_user_nonmember_index = np.arange(num_instance)[(this_user_idx + num_user) * eval_data_size:(this_user_idx + 1 + num_user) * eval_data_size]
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((eval_data_size)), np.zeros((eval_data_size))))
		else:
			this_user_member_index = np.arange(int(num_instance / 2))
			this_user_nonmember_index = np.setdiff1d(np.arange(num_instance), this_user_member_index)
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((len(this_user_member_index))), np.zeros((len(this_user_member_index)))))
		
		this_user_cos = all_epoch_layer_cos[this_user_index, this_user_idx, :, :]
		this_user_grad_diff = all_epoch_layer_grad_diff[this_user_index, this_user_idx, :, :]
		this_user_grad_norm = all_epoch_layer_grad_norm[this_user_index, this_user_idx, :, :]
		this_user_loss = all_epoch_loss[this_user_index, this_user_idx, :]
		#this_user_loss_reduction = all_epoch_target_loss[this_user_index, this_user_idx, :] - all_epoch_loss[this_user_index, this_user_idx, :]
		
		this_user_auc_list = []
		this_user_acc_list = []
		this_user_tpr_list = []
		
		### report layerwise AUC score for each sum
		for this_layer in [best_layer]:
			this_layer_cos = this_user_cos[:, :, this_layer]
			this_layer_grad_diff = this_user_grad_diff[:, :, this_layer]
			this_layer_loss = this_user_loss[:, :]
			this_layer_grad_norm = this_user_grad_norm[:, :, this_layer]
			if (this_user_idx < num_user):  # <10
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_cos, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, np.average(this_layer_cos,weights=cos_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_cos, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_grad_diff, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_grad_diff, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_loss, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, -1*np.average(this_layer_loss,weights=loss_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_loss, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_grad_norm, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_grad_norm, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * this_layer_loss[:, -1], pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*this_layer_loss[:, -1],true_label))
				all_user_tpr_list.append((fpr, tpr))
				
			cos_auc_score = roc_auc_score(true_label, np.sum(this_layer_cos, axis=1))
			grad_diff_auc_score = roc_auc_score(true_label, np.sum(this_layer_grad_diff, axis=1))
			loss_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_loss, axis=1))
			grad_norm_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_grad_norm, axis=1))
			last_epoch_loss_auc_score = roc_auc_score(true_label, -1 * this_layer_loss[:, -1])
			this_user_auc_list.append((cos_auc_score, grad_diff_auc_score, loss_auc_score, grad_norm_auc_score,last_epoch_loss_auc_score))
			
			cos_acc_score = report_acc(np.sum(this_layer_cos, axis=1), true_label)
			grad_diff_acc_score = report_acc(np.sum(this_layer_grad_diff, axis=1), true_label)
			loss_acc_score = report_acc(-1 * np.sum(this_layer_loss, axis=1), true_label)
			grad_norm_acc_score = report_acc(-1 * np.sum(this_layer_grad_norm, axis=1), true_label)
			last_epoch_loss_acc_score = report_acc(-1 * this_layer_loss[:, -1], true_label)
			this_user_acc_list.append((cos_acc_score, grad_diff_acc_score, loss_acc_score, grad_norm_acc_score,last_epoch_loss_acc_score))
		
		auc_list.append(this_user_auc_list)
		acc_list.append(this_user_acc_list)
		tpr_list.append(this_user_tpr_list)
	
	auc_list = np.array(auc_list)
	acc_list = np.array(acc_list)
	tpr_list = np.array(tpr_list)
	
	avg_auc = np.squeeze(np.average(auc_list, axis=0))
	avg_acc = np.squeeze(np.average(acc_list, axis=0))
	avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
	
	std_auc = np.squeeze(np.std(auc_list,axis=0))
	std_tpr = np.squeeze(np.std(tpr_list, axis=0))
	
	#print (f"current layer {best_layer}")
	
	print (f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[4]:.4f}")
	
	print(
		f"std over 10 users: cos AUC {std_auc[0]:.4f}, grad diff AUC {std_auc[1]:.4f}, loss AUC {std_auc[2]:.4f}, grad norm AUC {std_auc[3]:.4f}, last epoch loss AUC {std_auc[4]:.4f}")
	
	print (f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")
	
	print(
		f"std over 10 users at fpr =1e-3 : cos tpr {std_tpr[0]:.4f},  grad diff tpr {std_tpr[1]:.4f},  loss tpr {std_tpr[2]:.4f}, grad norm tpr {std_tpr[3]:.4f}, last epoch loss tpr {std_tpr[4]:.4f}")
	
	return_tpr = []
	for i in range(5):
		fpr, tpr = avg_roc_cal([all_user_roc_list[j * 5 + i] for j in range(num_user)])
		return_tpr.append((fpr,tpr))
	return_tpr = np.array(return_tpr)
	
	return avg_auc, avg_tpr,return_tpr


def all_epoch_analysis_all_adversary_auc_tpr_valid_based_rank(all_epoch_layer_cos, all_epoch_layer_grad_diff, all_epoch_loss, all_epoch_layer_grad_norm,
                                             label, best_layer, eval_data_size, num_user, f, valid_all_epoch_cos,valid_all_epoch_grad_diff,valid_all_epoch_grad_norm,
	                                                             valid_all_epoch_loss):
	auc_list = []
	acc_list = []
	tpr_list = []
	all_user_roc_list = []
	num_instance = all_epoch_layer_cos.shape[0]
	num_user = all_epoch_layer_cos.shape[1]
	num_layer = all_epoch_layer_cos.shape[3]
	num_epochs = all_epoch_layer_cos.shape[2]
	
	all_user_tpr_list = []
	
	for this_user_idx in range(num_user):
		
		if (num_user != 1):
			this_user_member_index = np.arange(num_instance)[this_user_idx * eval_data_size:(this_user_idx + 1) * eval_data_size]
			this_user_nonmember_index = np.arange(num_instance)[(this_user_idx + num_user) * eval_data_size:(this_user_idx + 1 + num_user) * eval_data_size]
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((eval_data_size)), np.zeros((eval_data_size))))
		else:
			this_user_member_index = np.arange(int(num_instance / 2))
			this_user_nonmember_index = np.setdiff1d(np.arange(num_instance), this_user_member_index)
			this_user_index = np.concatenate((this_user_member_index, this_user_nonmember_index))
			true_label = np.concatenate((np.ones((len(this_user_member_index))), np.zeros((len(this_user_member_index)))))
		
		this_user_cos = all_epoch_layer_cos[this_user_index, this_user_idx, :, :]
		this_user_grad_diff = all_epoch_layer_grad_diff[this_user_index, this_user_idx, :, :]
		this_user_grad_norm = all_epoch_layer_grad_norm[this_user_index, this_user_idx, :, :]
		this_user_loss = all_epoch_loss[this_user_index, this_user_idx, :]
		
		this_user_auc_list = []
		this_user_acc_list = []
		this_user_tpr_list = []
		
		### report layerwise AUC score for each sum
		for this_layer in [best_layer]:
			this_layer_cos = this_user_cos[:, :, this_layer]
			this_layer_grad_diff = this_user_grad_diff[:, :, this_layer]
			this_layer_loss = this_user_loss[:, :]
			this_layer_grad_norm = this_user_grad_norm[:, :, this_layer]
			valid_layer_cos = valid_all_epoch_cos[:,this_user_idx,:,this_layer]
			valid_layer_grad_diff = valid_all_epoch_grad_diff[:,this_user_idx,:,this_layer]
			valid_layer_loss = valid_all_epoch_loss[:,this_user_idx,:]
			valid_layer_grad_norm = valid_all_epoch_grad_norm[:,this_user_idx,:,this_layer]
			
			this_layer_cos = np.nan_to_num(this_layer_cos)
			this_layer_grad_diff = np.nan_to_num(this_layer_grad_diff)
			this_layer_loss = np.nan_to_num(this_layer_loss)
			this_layer_grad_norm = np.nan_to_num(this_layer_grad_norm)
			
			if (this_user_idx < num_user):  # <10
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_cos, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, np.average(this_layer_cos,weights=cos_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_cos, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, np.sum(this_layer_grad_diff, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(np.sum(this_layer_grad_diff, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_loss, axis=1), pos_label=1)
				#fpr, tpr, thresholds = metrics.roc_curve(true_label, -1*np.average(this_layer_loss,weights=loss_weights,axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_loss, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * np.sum(this_layer_grad_norm, axis=1), pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*np.sum(this_layer_grad_norm, axis=1),true_label))
				all_user_tpr_list.append((fpr, tpr))
				
				fpr, tpr, thresholds = metrics.roc_curve(true_label, -1 * this_layer_loss[:, -1], pos_label=1)
				all_user_roc_list.append(np.stack((fpr, tpr)))
				this_user_tpr_list.append(get_tpr(-1*this_layer_loss[:, -1],true_label))
				all_user_tpr_list.append((fpr, tpr))
			
			cos_auc_score = roc_auc_score(true_label, np.sum(this_layer_cos, axis=1))
			grad_diff_auc_score = roc_auc_score(true_label, np.sum(this_layer_grad_diff, axis=1))
			loss_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_loss, axis=1))
			grad_norm_auc_score = roc_auc_score(true_label, -1 * np.sum(this_layer_grad_norm, axis=1))
			last_epoch_loss_auc_score = roc_auc_score(true_label, -1 * this_layer_loss[:, -1])
			this_user_auc_list.append((cos_auc_score, grad_diff_auc_score, loss_auc_score, grad_norm_auc_score,last_epoch_loss_auc_score))
			
			cos_acc_score = report_acc(np.sum(this_layer_cos, axis=1), true_label)
			grad_diff_acc_score = report_acc(np.sum(this_layer_grad_diff, axis=1), true_label)
			loss_acc_score = report_acc(-1 * np.sum(this_layer_loss, axis=1), true_label)
			grad_norm_acc_score = report_acc(-1 * np.sum(this_layer_grad_norm, axis=1), true_label)
			last_epoch_loss_acc_score = report_acc(-1 * this_layer_loss[:, -1], true_label)
			this_user_acc_list.append((cos_acc_score, grad_diff_acc_score, loss_acc_score, grad_norm_acc_score,last_epoch_loss_acc_score))
		
		auc_list.append(this_user_auc_list)
		acc_list.append(this_user_acc_list)
		tpr_list.append(this_user_tpr_list)
	
	auc_list = np.array(auc_list)
	acc_list = np.array(acc_list)
	tpr_list = np.array(tpr_list)
	
	
	avg_auc = np.squeeze(np.average(auc_list, axis=0))
	avg_acc = np.squeeze(np.average(acc_list, axis=0))
	avg_tpr = np.squeeze(np.average(tpr_list, axis=0))
	
	#print(f"current layer {best_layer}")
	
	print(
		f"avg over 10 users: cos AUC {avg_auc[0]:.4f}, grad diff AUC {avg_auc[1]:.4f}, loss AUC {avg_auc[2]:.4f}, grad norm AUC {avg_auc[3]:.4f}, last epoch loss AUC {avg_auc[4]:.4f}")
	
	print(
		f"avg over 10 users: cos Acc {avg_acc[0]:.4f},  grad diff Acc {avg_acc[1]:.4f},  loss Acc {avg_acc[2]:.4f}, grad norm Acc {avg_acc[3]:.4f}, last epoch loss Acc {avg_acc[4]:.4f} ")
	
	print(
		f"avg over 10 users at fpr =1e-3 : cos tpr {avg_tpr[0]:.4f},  grad diff tpr {avg_tpr[1]:.4f},  loss tpr {avg_tpr[2]:.4f}, grad norm tpr {avg_tpr[3]:.4f}, last epoch loss tpr {avg_tpr[4]:.4f}")
	
	return_tpr = []
	for i in range(5):
		fpr, tpr = avg_roc_cal([all_user_roc_list[j * 5 + i] for j in range(num_user)])
		return_tpr.append((fpr, tpr))
	return_tpr = np.array(return_tpr)
	
	return avg_auc, avg_tpr, return_tpr


