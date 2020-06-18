import random
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms.functional as TF
from torch.utils import data

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

class ExpandedMNIST(data.Dataset):

	def __init__(self, root, train, transformation=None):
		
		# Can be train/test based on bool value of train
		dataset = datasets.MNIST(root=root, train=train, download=True)
		self.mnist_data = dataset.data
		self.mnist_labels = dataset.targets
		self.transform = transformation
		self.classes = list(range(10))
		print('Dataset Length for Expanded MNIST is ', len(dataset.data))
		print("Classes: ", self.classes)


	def __getitem__(self, index):
		
		img, label = self.mnist_data[index], self.mnist_labels[index]
		# np_img = img.numpy()
		img=img.numpy()
		if img.ndim==2:
			img = img.reshape(img.shape[0], img.shape[1], 1)
		
		expanded_img = place_random(img)
		tensor_img = TF.to_tensor(expanded_img)
		tensor_img = tensor_img.repeat(3, 1, 1)
		return tensor_img, label

	def __len__(self):
		return int(len(self.mnist_labels))

def place_random(img):
	#randomly place 28x28 mnist image on 40x40 background
	img_new = np.zeros((40,40,1), dtype=np.float32)
	x = np.random.randint(12 , size=1)[0]
	y = np.random.randint(12 , size=1)[0]

	img_new[y:y+28, x:x+28, :] = img
	
	return img_new


	 
