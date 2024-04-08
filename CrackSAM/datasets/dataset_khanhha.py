import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image


def random_rot_flip(image, label):
    # print(image.shape)  # 448,448,3
    # print(type(image))  # ndarray
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y,_ = image.shape
        # print(image.shape) #448,448,3
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y , 1), order=3) 
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))  
        sample = {'image': image, 'label': label>0.5} 

        return sample


class Khanhha_dataset(Dataset): 
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            filepath_image = self.data_dir + "images/{}".format(slice_name)
            filepath_label = self.data_dir + "masks/{}".format(slice_name) 
            image = np.array(Image.open(filepath_image))/255.0  # 448,448,3
            label = np.array(Image.open(filepath_label))/255.0    
        else: # test or val
            vol_name = self.sample_list[idx].strip('\n')
            filepath_image = self.data_dir + "images/{}".format(vol_name)
            filepath_label = self.data_dir + "masks/{}".format(vol_name)            
            image = np.array(Image.open(filepath_image))/255.0  # 448,448,3
            label = np.array(Image.open(filepath_label))/255.0  # 448,448

            image = torch.from_numpy(image) 
            label = torch.from_numpy(label)>0.5

        sample = {'image': image, 'label': label}
        if self.transform:  # train
            sample = self.transform(sample)  # torch 448,448,3
        sample['image'] = sample['image'].permute(2, 0, 1)  # torch 448,448,3  -> 3,448,448
        sample['case_name'] = self.sample_list[idx].strip('\n') 
        return sample




