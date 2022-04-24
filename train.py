# train process main script

import torch

from config import get_hyperparameters
from models import get_model
from logger import get_logger

def main():
    args = get_hyperparameters()

    for model_name in args['name']:

        model = get_model(model_name, args['n_class'])

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

        #train_loader = get_dataloader()
        #test_loader = get_dataloader()

        writer = get_logger(args['save_root'] )

if __name__ == '__main__':
    main()

