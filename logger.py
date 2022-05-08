from tensorboardX import SummaryWriter
import os
import torch
import copy 


def get_logger(save_root):
    writer = SaveManager(save_root)
    return writer

class SaveManager:
    def __init__(self, save_root):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok = True)

        self.writer = SummaryWriter(save_root)

        
        self.best_acc = None
        self.best_model = None

        self.result_summary_path = os.path.join(self.save_root, 'result_summary.txt')


    def add_scalar(self, text:str, value: float, global_step: int):
        self.writer.add_scalar(text, value, global_step)

    def add_image(self, text, image_grid, global_step):
        self.writer.add_image(text, image_grid, global_step)

    def update(self, model, acc):
        if self.best_acc is None:
            self.best_model = copy.deepcopy(model)
            self.best_acc = acc
            return

        if self.best_acc < acc:
            print('best_model update: {}'.format(acc))
            self.best_model = copy.deepcopy(model)
            self.best_acc = acc

    def save(self, model, prefix):
        checkpoint = {}
        checkpoint['weight'] = model.state_dict()

        save_path = os.path.join(self.save_root, str(prefix)+'.pt')
        torch.save(checkpoint, save_path)

    def close(self):

        self.writer.close()

        if self.best_model is not None:
            self.save(self.best_model, 'best')

        if os.path.isfile(self.result_summary_path):
            f_mode = 'a'
        else:
            f_mode = 'wt'

        if self.best_acc is not None:
            with open(self.result_summary_path, f_mode) as f:
                f.write('Best Scoire: '+str(self.best_acc)+'\n')
                print ('Best: Score: {}'.format(self.best_acc))