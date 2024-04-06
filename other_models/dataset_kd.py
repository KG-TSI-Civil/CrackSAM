import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image



class Khanhha_dataset_kd(Dataset):  # dataset for knowledge distillation
    def __init__(self, base_dir, list_dir):

        self.sample_list = open(os.path.join(list_dir, 'kd.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        vol_name = self.sample_list[idx].strip('\n')
        filepath_image = self.data_dir + "images/{}.jpg".format(vol_name)
        filepath_label = self.data_dir + "masks/{}.jpg".format(vol_name)

        # soft_label comes from the output of last layer of teacher
        filepath_soft_label = self.data_dir + "logits/{}.pt".format(vol_name)  
        
        # middle feature comes from the output of middle layer(s) of teacher, however, it's not recommended here
        filepath_soft_features = self.data_dir + "features/{}.pt".format(vol_name)      

        image = np.array(Image.open(filepath_image))/255.0  # 448,448,3
        label = np.array(Image.open(filepath_label))/255.0  # 448,448

        image = torch.from_numpy(image) # ndarray -> torch  #  448,448,3
        label = torch.from_numpy(label)>0.5

        logits = torch.load(filepath_soft_label, map_location='cpu')  #2,448,448
        features = torch.load(filepath_soft_features, map_location='cpu')  #256,28,28

        sample = {'image': image.float(), 'label': label.float() , 'soft_label':logits.float(), 'features': features.float()}

        sample['image'] = sample['image'].permute(2, 0, 1)  # torch 448,448,3  -> 3,448,448
        sample['case_name'] = self.sample_list[idx].strip('\n') 
        return sample



