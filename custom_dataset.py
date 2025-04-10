import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2  # Optional: for image loading
import numpy as np
from utils import resize_and_crop

class CustomDataset(Dataset):
    def __init__(self, pos_data_dir, neg_data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pos_data_dir = pos_data_dir
        self.neg_data_dir = neg_data_dir
        
        self.transform = transform
        
        self.pos_files = os.listdir(pos_data_dir)
        self.neg_files = os.listdir(neg_data_dir)

        self.pos_paths = [os.path.join(pos_data_dir, f) for f in self.pos_files]
        self.neg_paths = [os.path.join(neg_data_dir, f) for f in self.neg_files]

        self.data_files = self.pos_paths + self.neg_paths
        # shuffle the data
        np.random.shuffle(self.data_files)

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data_files)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to load.

        Returns:
            sample (dict): A dictionary containing the data and the label.
        """
        file_path = self.data_files[index]

        # Load image (modify this as per your data format)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        patches = resize_and_crop(image, target_size=(1904, 1120), patch_size=224)

        label = 1 if file_path in self.pos_paths else 0

        # Apply transformations if specified
        if self.transform is not None:
            patches = self.transform(patches)

        return patches, label