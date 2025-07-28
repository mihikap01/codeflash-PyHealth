from typing import Optional

import numpy as np
import torch
from torch import Tensor

from pyhealth.datasets import utils as datautils


def agg_loss(loss: torch.Tensor, reduction: str):
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def one_hot_np(labels, K):
    new_labels = np.zeros((len(labels), K))
    new_labels[np.arange(len(labels)), labels] = 1
    return new_labels


class LogLoss(torch.nn.Module):
    """Cross entropy, but takes in the probability instead of the logits"""

    reduction: str

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        clip=1e-10,
    ) -> None:
        super(LogLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.clip = clip

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimensions (got {dim})"
        input = input.clip(
            self.clip
        )  # this weight should be trivial, so I won't normalize
        input = -torch.log(input)
        if self.weight is not None:
            input = input * self.weight.unsqueeze(0)
        loss = torch.gather(input, -1, target.unsqueeze(-1)).squeeze(-1)
        return agg_loss(loss, self.reduction)


def prepare_numpy_dataset(
    model,
    dataset,
    keys,
    forward_kwargs=None,
    incl_data_keys=None,
    debug=False,
    batch_size=32,
):
    if forward_kwargs is None:
        forward_kwargs = {}
    if incl_data_keys is None:
        incl_data_keys = []
    loader = datautils.get_dataloader(dataset, batch_size, shuffle=False)

    ret = {k: [] for k in keys}
    if incl_data_keys:
        for k in incl_data_keys:
            ret[k] = []
    torch.set_grad_enabled(False)
    # tqdm can be slow on small or fast batches, so only enable if needed
    data_iter = loader
    if use_tqdm:
        import tqdm

        data_iter = tqdm.tqdm(
            enumerate(loader), desc=f"retrieving {keys}", total=len(loader)
        )
    else:
        data_iter = enumerate(loader)

    for _i, data in data_iter:
        if debug and (_i % 10 != 0):
            continue
        # Avoiding dict.update costs by constructing the kwargs dict flat
        data_combined = {**data, **forward_kwargs}
        res = model(**data_combined)
        for key in keys:
            out = res[key]
            ret[key].append(out.detach().cpu())
        for key in incl_data_keys:
            # This is NOT tensor from model, copy as is for efficiency
            ret[key].extend(data[key])

    # Efficient merging, only stack once per key
    for key in keys:
        ret[key] = _stack_to_numpy(ret[key])
    for key in incl_data_keys:
        ret[key] = np.asarray(ret[key])
    return ret


# Helper function to stack and convert to numpy efficiently
def _stack_to_numpy(tensors):
    if len(tensors) == 0:
        return np.array([])
    elif isinstance(tensors[0], np.ndarray):
        return np.concatenate(tensors, axis=0)
    else:
        return np.concatenate(
            [x.cpu().numpy() if torch.is_tensor(x) else np.array(x) for x in tensors],
            axis=0,
        )
