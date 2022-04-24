from tensorboardX import SummaryWriter
import os
import torch

def get_logger(save_root):
    writer = SaveManager(save_root)
    return writer

class SaveManager:
    def __init__(self, save_root):
        self.writer = SummaryWriter(save_root)

        self.best_acc = 0
        self.best_model = None

    def add_scalar(self, text, time_step, value):
        self.write.add_scalar(text, value, global_sept)

    def add_image(self, image, time_step):
        pass
