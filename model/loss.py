import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def weighted_cross_entropy_loss(output, target, weight):
    return F.cross_entropy(output, target, weight)
 
def categorical_cross_entropy_loss(output, target, axis=-1):
    # output (B, n_classes)
    # categorical_cross_entropy_loss is very different in keras
    # taken from keras implementation with torch backend
    # https://github.com/keras-team/keras/blob/v3.3.3/keras/src/backend/torch/nn.py
    output = output / torch.sum(output, dim=axis, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    output = torch.clip(output, eps, 1.0 - eps)
    log_prob = torch.log(output)
    # target supposed to be one_hot in keras, so needs to convert target to one_hot
    target = target.type(dtype=torch.long)
    target = F.one_hot(target, num_classes=output.shape[1])
    return torch.mean(-torch.sum(target * log_prob, dim=axis))


