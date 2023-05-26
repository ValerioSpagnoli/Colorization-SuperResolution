# In this file is defined the class of the dataset for PikFix Model

from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd
import torch

class PikFixData(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        '''
        Args:
            - csv_file: file csv of the format (org, ref, rst)
            - root_dir: root directory of the dataset
            - transforms: transformation to be applied to the data
        '''
        self.img_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get paths from i-th rows of csv
        org_img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        ref_img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 1])
        res_img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 2])

        # Read respective images
        org_img = read_image(org_img_path)
        ref_img = read_image(ref_img_path)
        res_img = read_image(res_img_path)
        
        # Transform them (if needed)
        if self.transform and not torch.is_tensor(org_img):
            org_img = self.transform(org_img)
            ref_img = self.transform(ref_img)
            res_img = self.transform(res_img)            

        # Return tuple (original, reference, restored)
        return org_img, ref_img, res_img