import copy

import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision
import cv2 as cv
import tensorflow as tf
from functools import partial
from torch.nn import functional as F


def _batchnorm_to_groupnorm_new(module):
    # print (module)
    # print (module.num_features)
    return nn.GroupNorm(num_groups=module.num_features, num_channels=module.num_features, affine=True)


def convert_model_from_pytorch_to_tensorflow(model, test_error=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
    input_var = Variable(torch.FloatTensor(input_np).to(device))
    output = model(input_var)

    k_model = pytorch_to_keras(model, input_var, (3, 32, 32,), verbose=False, change_ordering=True)

    # k_model.summary()

    return k_model


def sobel(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        ret[i, :] = gradxy
    return ret


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    '''
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
    """Adds a similarity loss term, the MMD between two representations.
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
      source_samples: a tensor of shape [num_samples, num_features].
      target_samples: a tensor of shape [num_samples, num_features].
      weight: the weight of the MMD loss.
      scope: optional name scope for summary tags.
    Returns:
      a scalar tensor representing the MMD loss value.
    """
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight

    return loss_value


def get_train_test_acc(user_list, target_model, print_option=True, return_loss=False, return_validation_result=False,return_ece_loss=False):
	#### get the train/test accuracy after training
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
	
	train_acc = 0
	test_acc = 0
	
	train_loss = 0
	test_loss = 0
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	idxs_users = len(user_list)
	
	correct = 0.0
	total = 0.0
	for idx in range(idxs_users):
		target_model.eval()
		for images, labels, _ in user_list[idx].train_data_loader:
			target_model.zero_grad()
			images = images.to(device)
			outputs = target_model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
			
			loss = criterion(outputs, labels).detach().item()
			train_loss += loss
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0
	
	train_acc = acc
	train_loss = train_loss / total
	
	if (print_option):
		print ("training accuracy %.2f" % (acc))
	
	
	correct = 0.0
	total = 0.0
	ece_criterion = _ECELoss().cuda()
	
	logits_list = []
	labels_list = []
	
	for images, labels, _ in user_list[idx].test_data_loader:
		target_model.zero_grad()
		images = images.to(device)
		outputs = target_model(images)
		labels = labels.to(device)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
		
		loss = criterion(outputs, labels).detach().item()
		test_loss += loss
		
		logits_list.append(outputs.detach())
		labels_list.append(labels.detach())
	
	logits = torch.cat(logits_list).cuda()
	labels = torch.cat(labels_list).cuda()
	ece_loss = ece_criterion(logits, labels).detach().item()
	
	acc = correct.item()
	acc = acc / total
	acc = acc * 100.0

	test_acc = acc
	test_loss = test_loss / total

	if (print_option):
		print ("testing accuracy %.2f" % (acc))

	if (return_validation_result):
		val_acc = 0
		val_loss = 0
	
		correct = 0.0
		total = 0.0
		target_model.eval()
		for images, labels, _ in user_list[0].validation_data_loader:
			target_model.zero_grad()
			images = images.to(device)
			outputs = target_model(images)
			labels = labels.to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
		
			loss = criterion(outputs, labels).detach().item()
			val_loss += loss
	
		acc = correct.item()
		acc = acc / total
		acc = acc * 100.0
	
		val_acc = acc
		val_loss = val_loss / total
	
		return train_acc,test_acc,val_acc,train_loss,test_loss,val_loss

	if (return_loss):
		# print (train_acc,test_acc,train_loss,test_loss)
		if (return_ece_loss):
			return train_acc, test_acc, train_loss, test_loss,ece_loss
		else:
			return train_acc, test_acc, train_loss, test_loss

	return train_acc, test_acc


def calculate_volume(model_weight_list, prev_weight_list, lr=0 , num_step=1, max_dim=None):
	
	#print (len(model_weight_list),len(prev_weight_list))
	
	flattened_model_param = []
	for this_model_param in model_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
			# print (val.size())
		cat_all_tensors = torch.cat(all_tensors)
		# print (cat_all_tensors.size())
		flattened_model_param.append(cat_all_tensors)
		
	prev_flattened_model_param = []
	for this_model_param in prev_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		cat_all_tensors = torch.cat(all_tensors)
		prev_flattened_model_param.append(cat_all_tensors)
		
	#print (len(flattened_model_param),len(prev_flattened_model_param))
		
	if (len(prev_flattened_model_param) == 1):
		### learning rate adjustment
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[0] + (
					flattened_model_param[idx] - prev_flattened_model_param[0]) / (lr) ### should we add num step here? so we are actually calculating
	else:
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[idx] + (flattened_model_param[idx] - prev_flattened_model_param[idx]) / (lr)
	
	
	matrix_a = torch.stack(flattened_model_param)
	#print (matrix_a.size()) ### this should be # of points * # of params
	matrix_a = matrix_a.cpu().numpy()  #### gpu memory is not enough to do SVD
	
	from sklearn.decomposition import PCA
	#print (f"n_components {matrix_a.shape[0]-1}")
	if (max_dim!=None):
		pca = PCA(n_components=max_dim)
	else:
		pca = PCA(n_components=matrix_a.shape[0]-1)
	points = pca.fit_transform(matrix_a)
	#print (points.shape)
	#print (pca.singular_values_)
	
	from scipy.spatial import ConvexHull, convex_hull_plot_2d
	hull = ConvexHull(points)
	#print (hull.area,hull.volume)
	return (hull.area, hull.volume)


def flatten_weight_dict(weight):
    all_tensors = []
    for key, val in weight.items():
        all_tensors.append(val.flatten())
        # print (val.size())
    cat_all_tensors = torch.cat(all_tensors)
    return cat_all_tensors


def calculate_param_cosine_similarity(param1, param2, param3):
    # idx = 0
    # for (key1,val1),(key2,val2),(key3,val3) in zip(param1.items(),param2.items(),param3.items()):
    #    if (idx == best_layer):
    #        layer1 = val1
    #        layer2 = val2
    #    idx+=1

    param1 = flatten_weight_dict(param1)
    param2 = flatten_weight_dict(param2)
    param3 = flatten_weight_dict(param3)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    
    ## gradient implementation
    cos_result = cos(torch.flatten(param1), torch.flatten(param2) - torch.flatten(param3))
    
    ## optim implementation
    # cos_result = cos(torch.flatten(param1)-torch.flatten(param3),torch.flatten(param2)-torch.flatten(param3))
    # cos_result = cos(torch.flatten(layer1),torch.flatten(layer2))
    return cos_result


def calculate_param_change_magnitude(weight_before, weight_after):
    flattened_weight_before = flatten_weight_dict(weight_before)
    flattened_weight_after = flatten_weight_dict(weight_after)

    # print (flattened_weight_after.size())
    # print (torch.sum(torch.abs(flattened_weight_after)))
    # print (torch.norm(flattened_weight_after,p=1))

    magnitude = torch.norm((flattened_weight_before - flattened_weight_after), p=1)

    return magnitude


def volume_dim_test(model_weight_list,prev_weight_list,lr=0):
	
	volume_dim_result = []
	
	flattened_model_param = []
	for this_model_param in model_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		# print (val.size())
		cat_all_tensors = torch.cat(all_tensors)
		# print (cat_all_tensors.size())
		flattened_model_param.append(cat_all_tensors)
	
	prev_flattened_model_param = []
	for this_model_param in prev_weight_list:
		all_tensors = []
		for key, val in this_model_param.items():
			all_tensors.append(val.flatten())
		cat_all_tensors = torch.cat(all_tensors)
		prev_flattened_model_param.append(cat_all_tensors)
	
	print(len(flattened_model_param), len(prev_flattened_model_param))
	
	if (len(prev_flattened_model_param) == 1):
		### learning rate adjustment
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[0] + (
					flattened_model_param[idx] - prev_flattened_model_param[0]) / (
				                             lr)  ### should we add num step here? so we are actually calculating
	else:
		for idx in range(len(flattened_model_param)):
			flattened_model_param[idx] = prev_flattened_model_param[idx] + (
						flattened_model_param[idx] - prev_flattened_model_param[idx]) / (lr)
	
	matrix_a = torch.stack(flattened_model_param)
	# print (matrix_a.size()) ### this should be # of points * # of params
	matrix_a = matrix_a.cpu().numpy()  #### gpu memory is not enough to do SVD
	matrix_a = np.nan_to_num(matrix_a)
	from sklearn.decomposition import PCA
	
	num_points = matrix_a.shape[0]
	print (f"total user:{num_points}")
	for dim in range(2,num_points):
		print (f"current dim {dim}")
		pca = PCA(n_components=dim)
		points = pca.fit_transform(copy.deepcopy(matrix_a))
		from scipy.spatial import ConvexHull, convex_hull_plot_2d
		hull = ConvexHull(points)
		volume_dim_result.append((hull.area,hull.volume))
		print (hull.area,hull.volume)
	return volume_dim_result



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece