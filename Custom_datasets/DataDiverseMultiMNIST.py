import random
import torch
import numpy as np
from torchvision import datasets

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

class DiverseMultiMNIST(datasets.MNIST):

    def __init__(self, pad, shift, num_pairs, root, train, transformation=None):
        super(SimpleMNIST, self).__init__(root, train, download=True)
        
        # Can be train/test based on bool value of train
        dataset = datasets.MNIST(root=root, train=train, download=True)
        self.pad = pad
        self.shift=shift
        self.mnist_data = dataset.data
        self.mnist_labels = dataset.targets
        self.transform = transformation
        self.classes = list(range(10))
        self.num_pairs=num_pairs
        print('Dataset Length is ', len(dataset.data))
        print("Classes: ", self.classes)


    def __getitem__(self, index):
        
        img, label = self.mnist_data[index], self.mnist_labels[index]
        np_img = img.numpy()
        random_shifts = np.random.randint(-self.shift, self.shift + 1,2)
        padded_image = np.pad(np_img, self.pad, 'constant')
        base_shifted = shift_2d(padded_image, random_shifts, shift)

        



        for i in np.arange(-self.shift, self.shift + 1):
            for j in np.arange(-self.shift, self.shift + 1):
                image_raw = shift_2d(padded_image, (i, j), self.shift)   
        if image_raw.ndim==2:
            image_raw = image_raw.reshape(image_raw.shape[0], image_raw.shape[1], 1)
        tensor_img = TF.to_tensor(image_raw)
        tensor_img = tensor_img.repeat(3, 1, 1)
        return tensor_img, label










    def __getitem__(self, index):
        

        # Should be 36X36 due to padding
        img, label = self.mnist_data[index], self.mnist_labels[index]
        decision = np.random.choice(['SingleDigit', 'OverlappingDigit'], 1, p=[0.167, 0.833])
        print("decision: ", decision)
        if decision == 'SingleDigit':
            img, label = self.mnist_data[index], self.mnist_labels[index]
            if self.transform != None:
                img = self.transform(img)
            return img, label
        else:
            # Choose two classes at random
            new_list = self.classes[0,int(label)]+ self.classes[int(label)+1,:]
            print(new_list)
            digit_class = random.sample(new_list, 1)
            print("Chosen class:", digit_classes, " original: ", label)

            # Find all images corresonding to that class
            data = self.mnist_data[self.mnist_labels==digit_class]            
            second_img = data[int(np.random.choice(data.shape[0], 1)), :, :]
            
            
            return img, label



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
  shifted_image = rolled_image
  # shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
  return shifted_image




     
