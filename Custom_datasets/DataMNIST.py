import random
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms.functional as TF
random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

class SimpleMNIST(datasets.MNIST):

    def __init__(self, pad, shift, root, train, transformation=None):
        super(SimpleMNIST, self).__init__(root, train, download=True)
        
        # Can be train/test based on bool value of train
        dataset = datasets.MNIST(root=root, train=train, download=True)
        self.pad = pad
        self.shift=shift
        self.mnist_data = dataset.data
        self.mnist_labels = dataset.targets
        self.transform = transformation
        self.classes = list(range(10))
        print('Dataset Length is ', len(dataset.data))
        print("Classes: ", self.classes)


    def __getitem__(self, index):
        
        img, label = self.mnist_data[index], self.mnist_labels[index]
        np_img = img.numpy()
        padded_image = np.pad(np_img, self.pad, 'constant')
        i,j = np.random.randint(-self.shift, self.shift + 1, 2) 
        # print("Random ", i, " ", j)
        image_raw = shift_2d(padded_image, (i, j), self.shift)   
        if image_raw.ndim==2:
            image_raw = image_raw.reshape(image_raw.shape[0], image_raw.shape[1], 1)
        tensor_img = TF.to_tensor(image_raw)
        tensor_img = tensor_img.repeat(3, 1, 1)
        return tensor_img, label

def shift_2d(image, shift, max_shift):
  """Shifts the image along each axis by introducing zero.
  Args:
    image: A 2D numpy array to be shifted.
    shift: A tuple indicating the shift along each axis.
    max_shift: The maximum possible shift.
  Returns:
    A 2D numpy array with the same shape of image.
  """
  max_shift += 1
  padded_image = np.pad(image, max_shift, 'constant')
  rolled_image = np.roll(padded_image, shift[0], axis=0)
  rolled_image = np.roll(rolled_image, shift[1], axis=1)
  shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
  return shifted_image




     
