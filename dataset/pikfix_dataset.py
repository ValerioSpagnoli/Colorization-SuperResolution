# In this file is defined the class of the dataset for PikFix Model

from torch.utils.data import Dataset
from skimage import io
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import cv2


class PikFixData(Dataset):
    def __init__(self, csv_file, root_dir, transform, max_size=256):
        '''
        Return the PikFix dataset based on RealOld.\n
        Args:
            - csv_file: file csv of the format (org, ref, rst)
            - root_dir: root directory of the dataset
            - transforms: transformation to be applied to the data
            - max_size: the max size of resized image
        Return the PikFix dataset based on RealOld
        '''
        self.img_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_size = max_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get paths from i-th rows of csv
        org_img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        ref_img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 1])
        res_img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 2])

        # Read respective images
        org_img = io.imread(org_img_path)
        ref_img = io.imread(ref_img_path)
        res_img = io.imread(res_img_path)
        
        org_img = cv2.resize(org_img, dsize=resize(width=org_img.shape[1], height=org_img.shape[0], max_size=self.max_size))
        ref_img = cv2.resize(ref_img, dsize=resize(width=ref_img.shape[1], height=ref_img.shape[0], max_size=self.max_size))
        res_img = cv2.resize(res_img, dsize=resize(width=res_img.shape[1], height=res_img.shape[0], max_size=self.max_size))


        # Transform them (if needed)
        if self.transform is not None:
            org_img = self.transform(org_img)
            ref_img = self.transform(ref_img)
            res_img = self.transform(res_img)   

        # Return tuple (original, reference, restored)
        return org_img, ref_img, res_img
    

def resize(width=None, height=None, max_size=256):
    if width > height:
        new_width = max_size
        new_height = int(height*(new_width/width))
    else:
        new_height = max_size
        new_width = int(width * (new_height / height))

    return new_width, new_height