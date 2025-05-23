import torch
from torch.utils.data import Dataset

import os
import cv2  

from utils import resize_and_crop


class CustomDataset(Dataset):
    def __init__(self, pos_data_dir, neg_data_dir, transform=None, ret_img=False):
        
        self.pos_data_dir = pos_data_dir
        self.neg_data_dir = neg_data_dir
        
        self.transform = transform

        self.ret_img = ret_img
        
        self.pos_files = os.listdir(pos_data_dir)
        self.neg_files = os.listdir(neg_data_dir)

        self.pos_files = [f for f in self.pos_files if f.endswith('.jpg') or f.endswith('.png')]
        self.neg_files = [f for f in self.neg_files if f.endswith('.jpg') or f.endswith('.png')]

        self.pos_paths = [os.path.join(pos_data_dir, f) for f in self.pos_files]
        self.neg_paths = [os.path.join(neg_data_dir, f) for f in self.neg_files]

        self.data_files = self.pos_paths + self.neg_paths

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data_files)

    def __getitem__(self, index):
        file_path = self.data_files[index]

        # Load image (modify this as per your data format)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        patches = resize_and_crop(image, target_size=(1904, 1120), patch_size=224)
        # patches = resize_and_crop(image, target_size=(1920, 1088), patch_size=128)

        label = 1 if file_path in self.pos_paths else 0
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations if specified
        if self.transform is not None:
            patches = [self.transform(patch) for patch in patches]
            # Convert to tensor
            patches = torch.stack(patches)
        
        if self.ret_img:
            # Return the image as well
            return patches, label, image

        return patches, label