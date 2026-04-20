import os
import cv2
import torch
from torch.utils.data import Dataset
import config

class SegmentationDataset(Dataset):

    #Constructor with the paths to the images and masks 

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)


    #Returns the number of samples in the dataset
    def __len__(self):
        return len(self.images)
    

    # Returns the image and mask tensors for a given index
    def __getitem__(self, idx):

        name = self.images[idx]

        # Read the image and convert to a matrix
        img = cv2.imread(os.path.join(self.img_dir, name))
        # Convert BGR to RGB 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        img = img / 255.0

        # Convert .jpg to .png
        mask_name = os.path.splitext(name)[0] + ".png"
        # Read the mask as grayscale
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), 0)
        mask = cv2.resize(mask,(config.IMAGE_SIZE, config.IMAGE_SIZE),interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        img = torch.tensor(img).float().permute(2,0,1)
        mask = torch.tensor(mask).long()

        return img, mask