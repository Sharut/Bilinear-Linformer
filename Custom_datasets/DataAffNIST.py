import numpy as np
import scipy.io as sio
from glob import glob
import os
from torchvision import datasets
from torch.utils import data
import random
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms.functional as TF
from torch.utils import data


class AffNIST(data.Dataset):

	def __init__(self, root, train):
		# Can be train/test based on bool value of train
		if train:
			print("Loading train data...")
			affnist_dataset,  affnist_labels = load_train_affNIST(root_dir=root)
		else:
			print("loading test data...")
			affnist_dataset,  affnist_labels = load_test_affNIST(root_dir=root)

		self.affnist_data = affnist_dataset
		self.affnist_labels = affnist_labels
		print("Dataset length ", self.affnist_data.shape)


	def __getitem__(self, index):        
		img, label = self.affnist_data[index], self.affnist_labels[index]
		if img.ndim==2:
			img = img.reshape(img.shape[0], img.shape[1], 1)
		
		# print(type(img), type(label))
		tensor_img = TF.to_tensor(img)
		tensor_img = tensor_img.repeat(3, 1, 1)
		label = torch.from_numpy(np.array(label))
		return tensor_img, label

	def __len__(self):
		return int(len(self.affnist_data))


def load_data_from_mat(path):
	data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
	for key in data:
		if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
			data[key] = _todict(data[key])
	return data


def _todict(matobj):
	#A recursive function which constructs from matobjects nested dictionaries
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, sio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		else:
			dict[strg] = elem
	return dict


def load_test_affNIST(root_dir):
	train_path = glob(os.path.join(root_dir, "test/*.mat"))
	dataset = load_data_from_mat(train_path[0])


	ans_set = dataset['affNISTdata']['label_int']
	test_set = dataset['affNISTdata']['image']
	test_set = test_set.reshape((10000, 40, 40, 1))
	ans_set = ans_set.reshape((10000))

	print ('test_set',test_set.shape)# (10000, 40, 40, 1)
	print ('label_set',ans_set.shape)#(10000,)
	return test_set,ans_set

def load_train_affNIST(root_dir):
	# path = root_dir + '/affNIST/train/1.mat'
	train_path = glob(os.path.join(root_dir, "train/*.mat"))
	dataset = load_data_from_mat(train_path[0])

	ans_set = dataset['affNISTdata']['label_int']
	train_set = dataset['affNISTdata']['image']
	
	train_set = train_set.reshape((50000, 40, 40, 1))
	ans_set = ans_set.reshape((50000))
	

	print ('train_set',train_set.shape)# (50000, 40, 40, 1)
	print ('label_set',ans_set.shape)#(50000,)
	return train_set, ans_set


def main():
	dataset = AffNIST(root="../data/AffNIST", train=False)


# main()

