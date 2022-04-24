# Hyperparameter control
import os, shutil

def get_hyperparameters(config = None):
    if config is None: # Train
        _save()
    else: # test
        _load(config)

    ret = {}
    ret.update ( model_dict )
    ret.update ( data_dict )

    return ret

# model-related params
model_dict = dict(                  
    name = ['resnet18', 'vggnet16'],
    imagenet_pretrained = True,
    n_class = 4,
    max_epoch = 50,
    learning_rate = 1e-4,
    mile_stone = None,
    # mile_stone = [20, 40],
    decay_rate = 0.1,
    extra = ['etc', 'explain']
)

train_pipeline = [
    dict(
        type = 'Resize',
        width = 224,
        height = 224
    ),
    dict(
        type= 'ToTensor'
    ),
]
test_pipeline = [
    dict(
        type= 'ToTensor'
    ),
]


# dataset-related params
data_dict = dict(
    save_root = './work_dir',
    batch_size = 64,
    workers_per_gpu = 2,

    train = dict(
        img_dir = '/path/to/dataset',
        ann_file = './path/to/annotation',
        pipeline = train_pipeline
    ),
    test = dict(
        img_dir = '/path/to/dataset',
        ann_file = './path/to/annotation',
        pipeline = test_pipeline
    ),

)

def _save():
    model_version = []
    for k in ['name', 'imagenet_pretrained', 'extra']:
        if k in model_dict:
            if isinstance(model_dict[k], list):
                model_version += model_dict[k]
            else:
                model_version.append(str(model_dict[k]))
    
    os.makedirs(data_dict['save_root'], exist_ok = True)
    VERSION = '.'.join(model_version)
    VERSION = str('{:04d}'.format(len(os.listdir(data_dict['save_root'])) + 1) + '_') + VERSION

    SAVE_ROOT_DIR = os.path.join(data_dict['save_root'], VERSION)
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(SAVE_ROOT_DIR, __name__ + '.py'))
    data_dict['save_root'] = SAVE_ROOT_DIR

def _load():
    pass