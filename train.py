# train process main script

from config import get_hyperparameters
from models import get_model

def main():
    args = get_hyperparameters()

    for model_name in args['name']:

        model = get_model(model_name, args['n_class'])

        print (model)



if __name__ == '__main__':
    main()

