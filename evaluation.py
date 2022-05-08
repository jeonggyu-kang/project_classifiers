from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# TODO : ROC

def calc_accuracy(pred, gt):
    return accuracy_score(y_pred = pred.argmax(dim=1), y_true=gt)
