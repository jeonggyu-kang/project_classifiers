import os 
import copy 
import argparse 

from utils import parse_args_from_config 
from models import get_model 
from logger import get_logger 
from runner import tester 
from dataset import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default=None, help='Path to log directory')
    _args = parser.parse_args()

    if not os.path.exists(_args.log_root):
        print('Can not find directory {}'.format(_args.log_root))
        exit()


    _args.config = os.path.join(_args.log_root, 'config.py')
    modulevar = parse_args_from_config(_args.config)

    args = modulevar.get_hyperparameters(config=_args.config)


    for model_name in args['name']:              # loop in various models
        log_root = os.path.join(_args.log_root, model_name)

        ckpt_path = os.path.join(log_root, 'best.pt')
        model = get_model(model_name, args['n_class'], ckpt_path = ckpt_path)


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

        save_path = os.path.join(log_root, 'eval')
        writer = get_logger(save_path)


        tester(                        
            model = model,    
            test_loader = test_loader,
            writer = writer,
            hard_sample_mining = True,
            confusion_matrix = True
        )


if __name__ == '__main__':
    main()