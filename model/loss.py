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



def focal_loss(inputs, targets, gamma=2, alpha=None, reduction='mean'):
    """
    Focal Loss function (functional implementation).
    
    Args:
        inputs (Tensor): Predictions (logits) from the model.
        targets (Tensor): Ground truth labels.
        gamma (float): Focusing parameter that reduces the relative loss for well-classified examples.
        alpha (float or list): Balancing factor to address class imbalance (optional).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    
    Returns:
        Tensor: Computed focal loss.
    """
    # Convert inputs to probabilities using softmax (for multi-class) or sigmoid (for binary classification)
    if inputs.shape[1] > 1:  # multi-class classification
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
    else:  # binary classification
        probs = torch.sigmoid(inputs)
        targets_one_hot = targets.float().unsqueeze(1)

    # Calculate log probabilities
    log_probs = torch.log(probs + 1e-9)

    # Calculate the focal weight and apply it
    focal_weight = (1 - probs) ** gamma
    focal_loss = -focal_weight * targets_one_hot * log_probs

    # Apply alpha if specified (for class imbalance)
    if alpha is not None:
        if isinstance(alpha, (float, int)):  # single alpha for binary classification
            alpha_factor = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
        elif isinstance(alpha, (list, torch.Tensor)):  # class-wise alpha for multi-class
            alpha_factor = torch.tensor(alpha).to(inputs.device) * targets_one_hot
        focal_loss = alpha_factor * focal_loss

    # Apply the specified reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
