import numpy as np

from whitebox_attack import *
from blackbox_attack import *
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from model_utils import *
from opacus import PrivacyEngine
import gc

class User:
	def __init__(self,dataset,model_name,id):
		self.train_data = None
		self.train_label = None
		self.test_data = None
		self.test_label = None
		self.dataset = dataset
		self.train_dataset = None
		self.test_dataset = None
		self.train_eval_dataset = None
		self.test_eval_dataset = None
		#self.worker = sy.VirtualWorker(hook,id=id)
		self.train_data_loader = None
		self.test_data_loader = None
		self.train_eval_data_loader = None
		self.test_eval_data_loader = None
		self.train_gradient = None
		self.test_gradient = None
		self.train_grad_mean = None
		self.train_grad_var = None
		self.test_grad_mean = None
		self.test_grad_var = None
		self.train_loss_profile = []
		self.test_loss_profile = []
		self.optim = None
		self.model = None
		self.available_list = None
		self.target_transform = None
		self.train_transform = None
		self.test_transform = None
		self.class_weight = None
		self.scheduler = None
		self.valid_data = None
		self.valid_label = None
		self.validation_data_loader = None
		self.validation_data_set = None
		self.privacy_engine = None
	
	def create_new_train_data_loader(self,batch_size):
		
		if (self.available_list is None):
			self.available_list = np.arange(len(self.train_data))
		
		new_train_dataset = part_pytorch_dataset(self.train_data[self.available_list], self.train_label[self.available_list], train=True, transform=self.train_transform,
		                                         target_transform=self.target_transform)
		
		new_train_data_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size,
		                                                    shuffle=True, num_workers=1)
		
		return new_train_data_loader
	
	def update_ban_list(self,ban_list):
		
		#print ("original ",self.available_list)
		#print ("ban ",ban_list)
		
		self.available_list = np.setdiff1d(self.available_list,self.available_list[ban_list])
		if (len(self.available_list)==0):
			self.available_list = np.arange(len(self.train_data))
		#print ("new list length",len(self.available_list))
	
	def reset_ban_list(self):
		self.available_list = np.arange(len(self.train_data))
	
	
	def create_batch_attack_data_loader(self,data_index,batch_size):
		attack_dataset = part_pytorch_dataset(self.train_data[self.available_list[data_index]],self.train_label[self.available_list[data_index]],
		                                      train=False, transform=self.test_transform,target_transform=self.target_transform)
		attack_loader = torch.utils.data.DataLoader(attack_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
		return attack_loader

## utils for users

def update_weights(current_model_weights,model, optimizer, train_loader,local_epochs,mixup=0,num_step=10,class_weights=None,unequal=0,model_name=''):
	# Set mode to train model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	class_weights = torch.from_numpy(class_weights).type(torch.float32)
	#class_weights.type(torch.float32)
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device) #weight=class_weights
	model.train().to(device)
	model.load_state_dict(current_model_weights)
	model.zero_grad(set_to_none=True)
	#print (class_weights)
	
	#for param in model.parameters():
		#print (param.dtype)
		#break

	### for each user, we need to create a new dataloader, so we can avoid using instances that are used in previous steps, same epoch.
	### for single worker, it is not possible to play with sampler to achieve the above constraint.
	
	all_data_idx = []
	step_count = 0
	
	for iter in range(local_epochs):
		for batch_idx, (images, labels,data_idx) in enumerate(train_loader):
			optimizer.zero_grad()
			images, labels = images.to(device), labels.to(device)
			if (model_name == 'inception'):
				log_probs,_ = model(images)
				#print (batch_idx,images.shape)
			else:
				log_probs = model(images)
			loss = criterion(log_probs, labels)
			loss.backward()
			optimizer.step()
			all_data_idx.append(data_idx.detach().cpu().numpy())
			step_count+=1
			if (step_count == num_step and local_epochs==1 and (not unequal)):
				break
		
	model.zero_grad(set_to_none=True)
	all_data_idx = np.unique(np.array(all_data_idx).flatten())
	
	return all_data_idx


'''
def update_weights_dpsgd(current_model_weights,model, optimizer, train_loader, local_epochs,
                         mixup=0,batch_size=100,target_data_size=5000,noise_scale=1e-6,grad_norm=1e10):
    # Set mode to train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    model.load_state_dict(current_model_weights)
    model.zero_grad()
    ### adding dp components
    privacy_engine = PrivacyEngine(
        model,
        sample_rate = (batch_size/target_data_size),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), ### params for renyi dp
        noise_multiplier=noise_scale, ### sigma
        max_grad_norm=grad_norm) ### this is from dp-sgd paper)
    privacy_engine.attach(optimizer)

    for iter in range(local_epochs):
        if (int(mixup)==1):
            #print ('mixup is called!')
            for batch_idx, (images, labels,_) in enumerate(train_loader):
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, 1)## set mixup.alpha = 1
                optimizer.zero_grad()
                inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
                outputs = model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)
                loss.backward()
                optimizer.step()
        else:
            for batch_idx, (images, labels,_) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

    return model.state_dict()
'''

def update_weights_mmd(current_model_weights,model, optimizer, train_loader, test_loader,
                       local_epochs,mixup=0,batch_size=100,train_loader_in_order=None,
                       validation_set=None,loss_lambda=0.1,num_classes=100,starting_index=None,class_weights=None):
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	class_weights = torch.from_numpy(class_weights).type(torch.float32)
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
	
	#print (class_weights)
	
	model.train()
	model.load_state_dict(current_model_weights)
	model.zero_grad()
	
	for iter in range(local_epochs):
		if (int(mixup)==1):
			#print ('mixup is called!')
			for batch_idx, (images, labels,_) in enumerate(train_loader):
				inputs, targets_a, targets_b, lam = mixup_data(images, labels, 1)## set mixup.alpha = 1
				optimizer.zero_grad()
				inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
				outputs = model(inputs)
				loss_func = mixup_criterion(targets_a, targets_b, lam)
				loss = loss_func(criterion, outputs)
				loss.backward()
				optimizer.step()
		else:
			for batch_idx, (images, labels,_) in enumerate(train_loader):
				images, labels = images.to(device), labels.to(device)
				model.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels)
				loss.backward()
				optimizer.step()
		
		### after training of each epoch, we apply MMD loss here
		
		# get training accuracy
		
		correct = 0.0
		total = 0.0
		for images, labels,_ in train_loader:
			images = images.to(device)
			outputs = model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
		acc = correct.item()
		acc = acc / total
		acc = acc * 100.0
		this_training_acc = acc
		
		#print ('Training Accuracy %f' %(acc))
		
		# get validation accuracy
		
		correct = 0.0
		total = 0.0
		model.eval()
		for images, labels,_ in test_loader:
			images = images.to(device)
			outputs = model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
		acc = correct.item()
		acc = acc / total
		acc = acc * 100.0
		this_testing_acc = acc
		
		#print('Testing Accuracy %f ' % (acc))
		
		if (this_training_acc-this_testing_acc>3):
			#### the gap between train/test. the threshold here can be adjusted
			model.train()
			
			#validation_label_in_training = []
			#validation_confidence_in_training = []
			
			for loss_index, (train_images, train_labels,_) in enumerate(train_loader_in_order):
				model.zero_grad()
				### get the same number of images for each class
				valid_images = torch.zeros_like(train_images).type(torch.FloatTensor).to(device)
				valid_labels = torch.zeros_like(train_labels).type(torch.LongTensor).to(device)
				valid_index =0
				for label_index, i in enumerate(torch.unique(train_labels)):
					this_frequency = torch.bincount(train_labels)[i].to(device)
					this_class_start = starting_index[i]
					## i is the current class label
					
					if (i < num_classes - 1):
						this_class_end = starting_index[i + 1] - 1
					else:
						this_class_end = validation_set.__len__() - 1
					
					for i in range(this_frequency):
						random_index=np.random.randint(this_class_start, this_class_end)
						new_images, new_labels, _ = ((validation_set).__getitem__(random_index))
						valid_images[valid_index] = new_images.to(device)
						valid_labels[valid_index] = new_labels.to(device)
						valid_index+=1
				
				
				train_images = train_images.to(device)#.type(torch.float64)
				#train_labels = train_labels.to(device)
				outputs = model(train_images)
				all_train_outputs = F.softmax(outputs, dim=1)
				#all_train_outputs = all_train_outputs.view(-1, num_classes)
				#train_labels = train_labels.view(batch_num, 1)
				
				valid_images = valid_images.to(device)#.type(torch.float64)
				#valid_labels = valid_labels.to(device)
				outputs = model(valid_images)
				all_valid_outputs = F.softmax(outputs, dim=1)
				all_valid_outputs = (all_valid_outputs).detach_()
				#valid_labels = valid_labels.view(batch_num, 1)
				
				#validation_label_in_training.append(valid_labels.cpu().data.numpy())
				#validation_confidence_in_training.append(all_valid_outputs.cpu().data.numpy())
				
				mmd_loss = mix_rbf_mmd2(all_train_outputs, all_valid_outputs, sigma_list=[1]) * loss_lambda
				mmd_loss.backward()
				optimizer.step()
	
	return model.state_dict()



def average_weights(w,weight=None):
	"""
	Returns the average of the weights and the param diff from each user.
	"""
	#print (len(w))
	if (len(w) == 1):
		return w[0]
	if (weight is None):
		weight = torch.ones((len(w)))/len(w)
	else:
		weight = torch.from_numpy(weight)
	
	#print(weight)
	#w_avg = copy.deepcopy(w[0])
	w_avg = copy.deepcopy(w[0])
	
	for key,val in w_avg.items():
		w_avg[key] = val*weight[0]
	
	for key in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[key] += w[i][key]*weight[i]
			#w_avg[key] +=w[i][key]
		#w_avg[key] = torch.div(w_avg[key], len(w))
	return w_avg

def active_attacker_mislabel(model,optimizer,user_list,local_epochs=1,batch_size=100,client_adversary=0,lr_multiplier=0.1,target_label=0,num_classes=10):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
	model.train().to(device)
	# print (f"client adversary {client_adversary}, multiplier {lr_multiplier}")
	#print ("mislabel active attack")
	#print (f"target_label{target_label}")
	default_backup_target_label = 5
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_member_dataset, batch_size=50,
		                                     shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				
				if (target_label!=-1):
					### create wrong labels / target fix label
					temp_new_labels = torch.ones_like(labels)*target_label
					new_labels = torch.zeros_like(temp_new_labels)
					for idx,(old_label,new_label) in enumerate(zip(labels,temp_new_labels)):
						if (old_label == new_label):
							new_labels[idx] = default_backup_target_label
						else:
							new_labels[idx] = new_label
				else:
					### create wrong labels / target random label
					random_labels = torch.from_numpy(np.random.randint(0,num_classes,labels.size(0))).to(device)
					new_labels = torch.zeros_like(labels)
					for idx,(old_label,random_label) in enumerate(zip(labels,random_labels)):
						if (old_label == random_label):
							new_labels[idx] = (random_label+1)%num_classes
						else:
							new_labels[idx] = random_label
					
					#print (torch.unique(labels).size(),torch.unique(new_labels).size())
					#print (torch.unique(new_labels))
				
				model.zero_grad()
				log_probs = model(images)
				loss = -1 * lr_multiplier * criterion(log_probs, new_labels.to(device))
				loss.backward()
				optimizer.step()
				# print (batch_idx)
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_non_member_dataset, batch_size=50,
		                                     shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels, _) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				if (target_label!=-1):
					### create wrong labels / target fix label
					temp_new_labels = torch.ones_like(labels)*target_label
					new_labels = torch.zeros_like(temp_new_labels)
					for idx,(old_label,new_label) in enumerate(zip(labels,temp_new_labels)):
						if (old_label == new_label):
							new_labels[idx] = default_backup_target_label
						else:
							new_labels[idx] = new_label
				else:
					### create wrong labels / target random label
					random_labels = torch.from_numpy(np.random.randint(0,num_classes,labels.size(0))).to(device)
					new_labels = torch.zeros_like(labels)
					for idx,(old_label,random_label) in enumerate(zip(labels,random_labels)):
						if (old_label == random_label):
							new_labels[idx] = (random_label+1)%num_classes
						else:
							new_labels[idx] = random_label
				
				model.zero_grad()
				log_probs = model(images)
				loss = -1 * lr_multiplier * criterion(log_probs, new_labels)
				loss.backward()
				optimizer.step()
				# print (batch_idx)
	
	return model.state_dict()


def active_attacker_gradient_ascent(model, optimizer, user_list,local_epochs=1,ga_adversary=1,param_search=False,class_weights=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
	model.train().to(device)
	sum_loss = 0
	
	### if ga_adversary == 1, this means we are doing GA attack
	### if ga_adversary == -1, this means we are doing GD attack
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_member_dataset,batch_size=50,shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels,_) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				optimizer.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels) *(-1*ga_adversary)
				sum_loss+=loss.cpu().item()
				loss.backward()
				### before step, we need to check the previous param, current grad
				#print (model.fc1.weight[0,0])
				#print (model.fc1.weight.grad[0,0])
				optimizer.step()
				#print (batch_idx)
				#print (model.fc1.weight[0,0])
	
	for user_idx in range(len(user_list)):
		loader = torch.utils.data.DataLoader(user_list[user_idx].evaluation_non_member_dataset,batch_size=50,shuffle=False)
		for iter in range(local_epochs):
			for batch_idx, (images, labels,_) in enumerate(loader):
				images, labels = images.to(device), labels.to(device)
				optimizer.zero_grad()
				log_probs = model(images)
				loss = criterion(log_probs, labels) *(-1*ga_adversary)
				sum_loss+=loss.cpu().item()
				loss.backward()
				optimizer.step()
				#print (batch_idx)
	
	#print (f"sumloss{sum_loss}")
	sum_loss = sum_loss/((len(user_list))*2) ## caclculate the avg loss of targeted instances
	
	return model.state_dict(),sum_loss
	
	#for key,val in model.state_dict().items():
	#	print (key,val.size())
	#for name,param in model.named_parameters():
	#   print (name,param.size(),param.grad.size())
	#active_attacker_gradient = copy.deepcopy([param.grad for param in model.parameters()]) ## this implementation would occur errors for BN layer
	
	#print(active_grad_magnitude)
	#active_attacker_gradient_dict = {}
	#for idx,(key,val) in enumerate(model.state_dict().items()):
	#	active_attacker_gradient_dict[key] =
		
	#print (f"sumloss{sum_loss}")
	#sum_loss = sum_loss/((len(user_list))*2) ## caclculate the avg loss of targeted instances
	
	#active_grad_magnitude = 0
	#for this_grad in active_attacker_gradient:
		#print (this_grad.size())
	#	active_grad_magnitude+=torch.norm(torch.flatten(this_grad),p=1)
	#return active_attacker_gradient_dict,active_grad_magnitude,sum_loss

def get_user_update_list(model_dict,user_dict_list,learning_rate,num_batches):
	
	#for key,val in model_dict.items():
	#    print (key,val.size())
	#print (f"user update num batches {num_batches}, lr {learning_rate}")
	user_update_list = []
	for user_idx in range(len(user_dict_list)):
		this_user_update_list = []
		for param1,param2 in zip(model_dict.values(),user_dict_list[user_idx].values()):
			
			#if (user_idx == 0):
			#	print (param1.size(),param1.requires_grad,param2.size())
			#print (f"user update param shape{param1.size()}")
			
			#if (len(param1.shape)>1): ### avoid params from batchnorm / bias.
			this_user_update_list.append((param1 - param2)/(learning_rate*num_batches)) ### be careful with the order here, new-param = old-param - gradient
			### also, dividing by learning rate means each user update is sum of gradient for all batches
		user_update_list.append(this_user_update_list) ### this is the average gradient of all training samples. NEED to be scaled by # of instances when do Grad-diff.
		
		#if (user_idx == 0):
		#	print ("require grads list")
		#	for p in this_user_update_list:
		#		print (p.size())
		#	exit(0)
	
	return user_update_list

def simplex_uniform_sampling(num):
	
	sampling = np.zeros((num))
	
	# we sample num-1 numbers from (0,1), then add 0 and 1 to the array, so we have num+1 numbers. After sorting
	# the gap between each adjacent pairs is the probability we use.
	
	array = []
	for _ in range(num-1):
		#print (np.random.get_state())
		array.append(np.random.uniform(low=0,high=1))
	array.append(0)
	array.append(1)
	array = np.sort(np.array(array))
	for i in range(len(array)-1):
		sampling[i] = array[i+1] - array[i]
	
	#print (array,sampling,np.sum(sampling))
	
	return sampling


def user_update_list_sanity_check(user_list,user_update_list,get_gradient_func,model):
	
	### this is to check how different the user update is from the sum of gradients
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	
	all_info = []
	
	for user_idx in range(len(user_list)):
		
		print (f"for user {user_idx}:")
		
		this_user_info = []
		
		train_eval_dataset = user_list[user_idx].train_eval_dataset
		data_iter = torch.utils.data.DataLoader(train_eval_dataset, batch_size=100, shuffle=False,
		                                        num_workers=1)
		
		acc_batch_grad = []
		batch_count = 0
		for image,label,_ in data_iter:
			this_batch_grad = get_gradient_func(model,image.to(device),label.to(device))
			
			#for param in this_batch_grad:
			#    print (param.size())
			
			if (batch_count == 0):
				for param in this_batch_grad:
					acc_batch_grad.append(torch.zeros_like(param))
			
			for layer_idx in range(len(acc_batch_grad)):
				acc_batch_grad[layer_idx] +=this_batch_grad[layer_idx]
			
			batch_count+=1
		
		for layer_idx in range(len(acc_batch_grad)):
			acc_batch_grad[layer_idx] = acc_batch_grad[layer_idx]/batch_count
		
		print (f"total batch number {batch_count}")
		
		### now we compare user update and param
		
		for layer_idx,(param1,param2) in enumerate(zip(acc_batch_grad,user_update_list[user_idx])):
			param1 = torch.flatten(param1)
			param2 = torch.flatten(param2)
			
			print (f"for layer {layer_idx}:")
			
			print (f"norm of batch grad {torch.norm(param1,p=1).item()}")
			print (f"norm of user epoch grad {torch.norm(param2,p=1).item()}")
			print (f"norm of grad diff {torch.norm(param1-param2,p=1).item()}")
			print (f"cosine similarity {cos(param1,param2).item()}")
			
			this_user_info.append([torch.norm(param1,p=1).item(),
			                       torch.norm(param2,p=1).item(),
			                       torch.norm(param1-param2,p=1).item(),
			                       cos(param1,param2).item()])
		
		all_info.append(this_user_info)
	
	print (np.array(all_info).shape)
	
	return np.array(all_info)




