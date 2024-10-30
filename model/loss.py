import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def categorical_cross_entropy_loss(output, target, axis=-1):
    # categorical_cross_entropy_loss is very different in keras
    # taken from keras implementation with torch backend
    # https://github.com/keras-team/keras/blob/v3.3.3/keras/src/backend/torch/nn.py
    output = output / torch.sum(output, dim=axis, keepdim=True)
    output = torch.clip(output, 1e-8, 1.0 - 1e-8)
    log_prob = torch.log(output)
    return -torch.sum(target * log_prob, dim=axis)


 