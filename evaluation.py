
import itertools

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import torch 


# TODO : ROC

def calc_accuracy(pred, gt):
    return accuracy_score(y_pred = pred.argmax(dim=1), y_true=gt)

def renderer(func):
    def inner(*args, **kwargs):
        fig = func(*args, **kwargs)

        # rendering
        fig.canvas.draw()
        fig_arr = np.array(fig.canvas.renderer._renderer)
        fig_arr = cv2.cvtColor(fig_arr, cv2.COLOR_RGBA2RGB)

        fig_arr = fig_arr / 255
        fig_tensor = torch.from_numpy(fig_arr).permute(2,0,1)
        plt.close()

        return fig_tensor
    return inner

@renderer
def _plot_confusion_matrix(cm, normalize = True, labels=True, title='Confusion_matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2 

    if labels:
        for y, x in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(x, y, 
                "{:0.4f}".format(cm[y,x]),
                horizontalalignment="center", fontsize='xx-large', color="white" if cm[y, x] > thresh else "black")
            else:
                plt.text(x, y, "{:,}".format(cm[y, x]), horizontalalignment="center", fontsize='xx-large', color="white" if cm[y, x] > thresh else "black")

    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('True label', fontsize='xx-large')
    plt.xlabel('Predicted label\nACC={:0.4f} / miscls_rate={:0.4f}'.format(accuracy, misclass), fontsize='xx-large')

    return fig

def get_confusion_matrix_image(pred, gt, normalize=True):
    cm = confusion_matrix(y_pred=pred.argmax(1), y_true=gt)
    cm_tensor = _plot_confusion_matrix(cm, normalize=normalize, labels = True)

    return cm_tensor