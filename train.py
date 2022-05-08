# train process main script

import torch

from models import get_model                     # "models" is a directory 

from config import get_hyperparameters
from dataset import get_dataloader

from logger import get_logger
from runner import trainer


def main():
    # TODO : apply easydict
    args = get_hyperparameters()

    for model_name in args['name']:              # loop in various models

        model = get_model(model_name, args['n_class'])

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

        if args.get('mile_stone') is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones = args['mile_stone'], gamma = 0.1)            
        else:
            scheduler = None

        mode = 'train'

        train_loader = get_dataloader(
            data_dir = args[mode]['img_dir'],
            ann_path = args[mode]['ann_file'],
            mode = mode,
            batch_size = args['batch_size'],
            num_workers = args['workers_per_gpu'],
            pipeline = args[mode]['pipeline']
        )

        mode = 'test'
        test_loader = get_dataloader(
            data_dir = args[mode]['img_dir'],
            ann_path = args[mode]['ann_file'],
            mode = mode,
            batch_size = args['batch_size'],
            num_workers = args['workers_per_gpu'],
            pipeline = args[mode]['pipeline']
        )

        #writer = get_logger(args['save_root'] )
        writer = None


        trainer(                                      # from runner.py
            max_epoch = args['max_epoch'],
            model = model,
            train_loader = train_loader,
            test_loader = test_loader,
            loss_fn = loss,
            optimizer = optimizer,
            scheduler = scheduler,
            meta = {
                'save_every' : 10,
                'print_every' : 5,
                'test_every' : 10
            },
            writer = writer
        )


if __name__ == '__main__':
    main()

