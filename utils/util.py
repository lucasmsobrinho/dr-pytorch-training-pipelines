
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    # Confusion matrix: lines represent target values and columns
    # represent predicted values.
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.highest_class = 1
        self.confusion_matrix = np.zeros((2,2))
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_confusion_matrix(self, output, target):
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        for i in range(len(target)):
            target_class = int(target[i].item())
            print(f"target_class is {target_class}")
            output_class = int(pred[i].item())
            print(f"output_class is {output_class}")
            number = max(target_class, output_class)
            if number > self.highest_class:
                self.highest_class = number
                self.enlarge_confusion_matrix(number)

            self.confusion_matrix[target_class, output_class] += 1
            
        print(self.confusion_matrix)

    def enlarge_confusion_matrix(self, number):
        new_matrix = np.zeros((number+1,number+1))
        old_matrix_shape = self.confusion_matrix.shape[0]
        new_matrix[:old_matrix_shape,:old_matrix_shape] = self.confusion_matrix
        self.confusion_matrix = new_matrix
        print(new_matrix)

    def get_other_metrics(self):
        num_classes = self.confusion_matrix.shape[0]
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        for i in range(num_classes):
            if sum(self.confusion_matrix[:,i]) != 0:
                precision[i] = self.confusion_matrix[i, i] / sum(self.confusion_matrix[:,i])
            if sum(self.confusion_matrix[i,:]) != 0:
                recall[i] =  self.confusion_matrix[i, i] / sum(self.confusion_matrix[i,:])

        avg_precision = sum(precision) / num_classes
        avg_recall = sum(recall) / num_classes
        f1_score = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)

        self._data.loc['precision'] = [0,0,0]
        self._data.total['precision'] = sum(precision)            
        self._data.counts['precision'] = num_classes
        self._data.average['precision'] = avg_precision
        self._data.loc['recall'] = [0,0,0]
        self._data.total['recall'] = sum(recall)            
        self._data.counts['recall'] = num_classes            
        self._data.average['recall'] = avg_recall
        self._data.loc['f1_score'] = [0,0,0]
        self._data.average['f1_score'] = f1_score
        
    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
