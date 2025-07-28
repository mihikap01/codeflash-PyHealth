from typing import Union

import torch

from pyhealth.calib.utils import LogLoss

_MAX_KERNEL_VALUE = 1.


class RBFKernelMean(torch.nn.Module):
    def __init__(self, h=1.):
        super().__init__()
        self.h = h

    def forward(self, x, x1=None):
        #if x1 is None: return 1
        _, dim = x.shape
        if x1 is None:
            x1 = x
        d = torch.pow(torch.cdist(x,x1,2), 2) / (-dim * self.h)
        return torch.exp(d)
    def set_bandwidth(self, h: Union[int, float]):
        """Set the bandwidth"""
        self.h = h
    def get_bandwidth(self):
        return self.h

def KDE_classification(X: torch.Tensor, Y: torch.Tensor, kern:RBFKernelMean=None, X_pred: torch.Tensor=None,
                       weights: Union[torch.Tensor, float, int]=1., min_eps:float=1e-10, drop_max:bool=False):
    """KDE classifier.
    We will be using X and Y to estimate the density for each of the K classes.
    kern is the kernel. X_pred is the data to predict.

    Args:
        X (torch.Tensor): Data of shape (N, d)
        Y (torch.Tensor): One-hot label of shape (N, K)
        kern (RBFKernelMean, optional): kernel.
            Defaults to None (a RBFKernelMean of bandwidth=1).
        X_pred (torch.Tensor, optional): Data to predict.
            Defaults to None, which means we will perform leave-one-out prediction.
        weights (Union[torch.Tensor, float, int], optional): Weights on each data in X.
            Defaults to 1.
        drop_max (bool, optional): Whether to ignore x where K(x_0, x) = 1.
            This typically means x is just x_0.
            Defaults to False.
            If you know there are overlap between X and X_pred, you could
            set this to True for convenience.
            Note that if X_pred=None (LOO) this is automatically set to True.

    Returns:
        torch.Tensor: Probability predictions for X_pred.
    """
    # kern, X, Y: assumed to be on same device and proper dtype
    if kern is None:
        kern = RBFKernelMean(h=1.)

    # Handle X_pred None (leave-one-out prediction)
    if X_pred is None:
        Kijs = kern(X, X)
        drop_max = True
        X_pred = X
    else:
        if X_pred.dim() == 1:
            X_pred = X_pred.unsqueeze(0)
        Kijs = kern(X_pred, X)

    # Efficient masking: avoid unnecessary where + broadcasting
    if drop_max:
        mask = Kijs >= (_MAX_KERNEL_VALUE - min_eps)
        if mask.any():
            Kijs = Kijs.masked_fill(mask, 0)

    # weights: support scalar or broadcasting to all columns
    if not torch.is_tensor(weights):
        if weights != 1.:
            Kijs = Kijs * weights
    else:
        Kijs = Kijs * weights

    # Efficient normalization: sum and safe division, in-place clamp
    norm = Kijs.sum(dim=1, keepdim=True)
    norm.clamp_(min=min_eps)
    Kijs = Kijs / norm

    # Efficient matrix multiply for predictions (may use out param in new PyTorch)
    pred = torch.matmul(Kijs, Y)
    return pred

def batched_KDE_classification(X: torch.Tensor, Y: torch.Tensor, kern=None, X_pred: torch.Tensor=None,
                       weights: Union[torch.Tensor, float, int]=1., min_eps=1e-10):
    # Use batches for large X_pred
    if X_pred is None:
        drop_max = True
        X_pred = X
    else:
        drop_max = False
    batch_size = 32
    n = X_pred.shape[0]
    outputs = []
    with torch.no_grad():
        for st in range(0, n, batch_size):
            ed = min(n, st + batch_size)
            outputs.append(
                KDE_classification(
                    X, Y, kern, X_pred[st:ed], weights, min_eps=min_eps, drop_max=drop_max
                )
            )
    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

class KDECrossEntropyLoss(torch.nn.Module):
    reduction: str
    def __init__(self, ignore_index: int = -100,
                 reduction: str = 'mean', h: float = 1.0, nclass=None) -> None:
        super(KDECrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.kern = RBFKernelMean(h=h)
        self.log_loss = LogLoss(reduction=reduction)
        self.nclass = nclass

    def forward(self, input: dict, target: torch.Tensor, eval_only=False) -> torch.Tensor:
        weights = input['weights'] if 'weights' in input else 1.
        # Optimized: avoid repeated .max() calls, precompute nclass, and use proper dtype
        nclass = self.nclass
        if nclass is None:
            # Only do the expensive .max() operations once if needed
            nclass = int(torch.maximum(input['supp_target'].max(), target.max()).item()) + 1
        supp_Y = torch.nn.functional.one_hot(input['supp_target'], nclass).float()

        if eval_only:
            pred = batched_KDE_classification(
                input['supp_embed'], supp_Y, self.kern, weights=weights, X_pred=input['pred_embed'])
        else:
            pred = KDE_classification(
                input['supp_embed'], supp_Y, self.kern, weights=weights, X_pred=input['pred_embed'])
        ret = {}
        ret['loss'] = self.log_loss(pred, target)
        ret['extra_output'] = {"prediction": pred}
        return ret
