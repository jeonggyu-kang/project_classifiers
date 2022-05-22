import os
import sys

import numpy as np
import cv2
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from utils import build_transform_from_cfg




def get_dataloader(dataset, data_dir, ann_path, mode, pipeline, batch_size, num_workers):
    if mode == 'train':
        is_training = True
        shuffle = True
    elif mode == 'test':
        is_training = False
        shuffle = False
    elif mode == 'valid':
        is_training = False
        shuffle = False
    else:
        raise ValueError

    df = pd.read_parquet(ann_path, engine='pyarrow')

    transform = build_transform_from_cfg(pipeline)

    #dataset = CoronaryArteryDataset(df, data_dir, transform, is_training = is_training)
    dataset = getattr(sys.modules[__name__], dataset)(df, data_dir, transform, is_training = is_training)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers=num_workers,
        drop_last = False
    )

    return dataloader



class CoronaryArteryDataset(Dataset):
    def __init__(self, df, data_dir, transform, is_training=False):
        self.is_training = is_training
        self.data = df.values

        self.data_dir = data_dir
        self.transform = transform

    def init_transforms(self,):
        raise NotImplementedError('This module has been deprected.')

        if self.is_training:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomAffine(
                        degrees=15, translate=(0.2, 0.2),
                        scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
                ]),
                transforms.ToTensor(),
                ])
        else:
            # val, test
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dcm_path,  score = self.data[index]
        
        # array_path = os.path.join('../Data/resized_224/',dcm_path.replace('.dcm','.npy'))
        array_path = os.path.join(self.data_dir,dcm_path.replace('.dcm','.npy'))

        image = np.load(array_path)

        # normalization (raw -> 255)
        image = image - image.min()
        image = image / image.max()

        # image = Image.fromarray(image.astype('uint8')).convert("RGB")
        transformed = self.transform({'image':image})
        image = transformed['image']

        score = torch.LongTensor([score]).squeeze()

        image = torch.cat([image,image, image], dim = 0).to(torch.float32)

        return image, score



class AGEDataset(Dataset):

    def __init__(self, df, data_dir, transform, is_training=False):
        super(AGEDataset, self).__init__()
        self.data = df.values
        self.transform = transform
        self.is_training = is_training

        self.data_dir = data_dir


        # TODO: modify
        self.range_min = 1
        self.range_max = 100
        self.range_max -= self.range_min


    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        dcm_path,  score = self.data[idx] # score dynamic range (600-3000)
        score = (score - self.range_min) / self.range_max


        array_path = os.path.join(self.data_dir,dcm_path.replace('.dcm','.npy'))
        image = np.load(array_path)

        image = image - image.min()
        image = image / image.max()

        transformed = self.transform({'image':image})
        image = transformed['image']
        x = torch.cat([image,image, image], dim = 0).to(torch.float32)


        y = torch.FloatTensor([score])

        return x, y
    
if __name__ == '__main__':
    pass