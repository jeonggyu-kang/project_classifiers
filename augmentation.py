import os
import cv2
import numpy as np

import torch

from models import get_model                     # "models" is a directory 

from config import get_hyperparameters            
from dataset import get_dataloader

from logger import get_logger
from runner import trainer


def main():
    # TODO : apply easydict
    args = get_hyperparameters()                 #             

    vanilla_pipeline = [
        dict(
            type = 'Resize',
            width = 896,
            height = 896
        ),

        dict(
            type= 'ToTensor'

        ),
    ] 

    pipeline = [
        dict(
            type = 'Resize',
            width = 896,
            height = 896
        ),
        dict(
            type = 'Contrastive',
            p = 1.0,
            w = 1.6

        ),
        dict(
            type= 'ToTensor'

        ),
    ]   
    
    mode = 'train'

    vanilla_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = vanilla_pipeline,
        # shuffle = False,
        csv = False
    )

    augmentation_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = pipeline,
        # shuffle = False,
        csv = False
    )


    for idx, (original, augmented) in enumerate(zip(vanilla_loader, augmentation_loader)):
        org_img, _ = original
        aug_img, _ = augmented

        np_org_img = (org_img[0]*255.0).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)
        np_aug_img = (aug_img[0]*255.0).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)

        org_file_name = 'original_{}.png'.format(idx+1)
        aug_file_name = 'augmentation_{}.png'.format(idx+1)

        cv2.imwrite(org_file_name, np_org_img)
        cv2.imwrite(aug_file_name, np_aug_img)

        if idx == 10:
            break


if __name__ == '__main__':
    main()