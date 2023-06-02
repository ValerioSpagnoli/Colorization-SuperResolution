# In this file is defined the class of the dataset for PikFix Model

from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms


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

        org_img = Image.open(org_img_path)
        ref_img = Image.open(ref_img_path)
        res_img = Image.open(res_img_path)

        width = org_img.size[0]
        height = org_img.size[1]

        org_img = org_img.resize(resize_to_maxsize(width=width, height=height, max_size=self.max_size))
        ref_img = ref_img.resize(resize_to_maxsize(width=width, height=height, max_size=self.max_size))
        res_img = res_img.resize(resize_to_maxsize(width=width, height=height, max_size=self.max_size))

        # org_img = org_img.resize(resize_to_maxsize(width=org_img.size[0], height=org_img.size[1], max_size=self.max_size))
        # ref_img = ref_img.resize(resize_to_maxsize(width=ref_img.size[0], height=ref_img.size[1], max_size=self.max_size))
        # res_img = res_img.resize(resize_to_maxsize(width=res_img.size[0], height=res_img.size[1], max_size=self.max_size))

        # Resise with heigh = width
        #org_img = org_img.resize((self.max_size,self.max_size))
        #ref_img = ref_img.resize((self.max_size,self.max_size))
        #res_img = res_img.resize((self.max_size,self.max_size))


        # Transform them (if needed)
        if self.transform is not None:
            
            org_img = self.transform(org_img)
            ref_img = self.transform(ref_img)
            res_img = self.transform(res_img)   


        # Return tuple (original, reference, restored)
        return (org_img, ref_img, res_img)
    

def resize_to_maxsize(width=None, height=None, max_size=256):
    if width > height:
        new_width = max_size
        new_height = int(height*(new_width/width))
        new_height = new_height - (new_height%8)

    else:
        new_height = max_size
        new_width = int(width * (new_height / height))
        new_width = new_width - (new_width%8)

    return new_width, new_height