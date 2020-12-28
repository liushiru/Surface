import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config as config


class SurfaceDataset(Dataset):

    default_tranform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if transform is None:
            self.transform = self.default_tranform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # to overwrite
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        surface = self.dataframe.iloc[idx, 1:].to_numpy()
        surface = surface.astype('float32')
        sample = {'image': image, 'surface': surface}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample




