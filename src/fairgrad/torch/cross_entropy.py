import warnings

from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

import torch
import numpy as np
from torch import Tensor
import numpy.typing as npt
from fairgrad.fairness_functions import *
from typing import Callable, Optional, Union


def get_fairness_function(fairness_function, all_train_y, all_train_s):
    fairness_setup_arguments = FairnessSetupArguments(
        fairness_function_name=fairness_function,
        y_unique=np.unique(all_train_y),
        s_unique=np.unique(all_train_s),
        all_train_y=all_train_y,
        all_train_s=all_train_s,
        y_desirable=np.asarray([1]),
    )

    return setup_fairness_function(fairness_setup_arguments)


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)


class CrossEntropyLoss(_WeightedLoss):
    r"""This is an extension of the CrossEntropyLoss provided by pytorch. Please check pytorch documentation
    for understanding the cross entropy loss. Here, we augment the cross entropy to enforce fairness. The
    exact algorithm can be found in Fairgrad paper <https://arxiv.org/abs/2206.10923>.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        y_train (np.asarray[int], Tensor, optional): All train example's corresponding label
        s_train (np.asarray[int], Tensor, optional): All train example's corresponding sensitive attribute. This means if there
            are 2 sensitive attributes, with each of them being binary. For instance gender - (male and female) and
            age (above 45, below 45). Total unique sentive attributes are 4.
        fairness_measure (string): Currently we support "equal_odds", "equal_opportunity", and "accuracy_parity".
        epsilon (float, optional): The slack which is allowed for the final fairness level.
        fairness_rate (float, optional): Parameter which intertwines current fairness weights with sum of previous fairness rates.



    Examples::

        >>> input = torch.randn(10, 5, requires_grad=True)
        >>> target = torch.empty(10, dtype=torch.long).random_(2)
        >>> s = torch.empty(10, dtype=torch.long).random_(2) # protected attribute
        >>> loss = CrossEntropyLoss(y_train = target, s_train = s, fairness_measure = 'equal_odds')
        >>> output = loss(input, target, s, mode='train')
        >>> output.backward()
    """
    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        y_train: Optional[Union[npt.NDArray[int], Tensor]] = None,
        s_train: Optional[Union[npt.NDArray[int], Tensor]] = None,
        fairness_measure: Optional[str] = None,
        epsilon: Optional[float] = 0.0,
        fairness_rate: Optional[float] = 0.01,
    ) -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        if y_train is not None and s_train is not None and fairness_measure is not None:
            self.fairness_flag = True
        else:
            self.fairness_flag = False
            warnings.warn(
                "Fairgrad mechanism is not employed as either y_train or s_train "
                "or fairness_measure is missing. Reverting to regular Cross Entropy Loss"
            )
        if self.fairness_flag:
            # Below keys is necessary
            assert all(y_train >= 0), "labels space must start with 0, and not -1"
            assert all(
                s_train >= 0
            ), "protected label space must start with 0, and not -1"
            self.y_train = y_train
            self.s_train = s_train
            self.epsilon = epsilon
            self.fairness_measure = fairness_measure
            self.fairness_rate = fairness_rate

            if type(self.y_train) == torch.Tensor:
                self.y_train = self.y_train.cpu().detach().numpy()
            if type(self.s_train) == torch.Tensor:
                self.s_train = self.s_train.cpu().detach().numpy()

            self.y_unique = np.unique(self.y_train)
            self.s_unique = np.unique(self.s_train)
            self.fairness_function = get_fairness_function(
                self.fairness_measure, self.y_train, self.s_train
            )

            # some lists to store all information over time. This in the future would be given as an option so
            # as to save space. @TODO: describe them individually.
            self.step_weights = []
            self.step_loss = []
            self.step_weighted_loss = []
            self.step_accuracy = []
            self.step_fairness = []
            self.cumulative_fairness = torch.zeros(
                (self.y_unique.shape[0], 2 * self.s_unique.shape[0])
            )

    def forward(
        self, input: Tensor, target: Tensor, s: Tensor = None, mode: str = "train"
    ) -> Tensor:
        loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        if self.fairness_flag and mode == "train":

            groupwise_fairness = self.fairness_function.groupwise(
                input.cpu().detach().numpy(),
                target.cpu().detach().numpy(),
                s.cpu().detach().numpy(),
            )

            if self.step_fairness != []:
                groupwise_fairness = np.where(
                    np.isnan(groupwise_fairness),
                    self.step_fairness[-1],
                    groupwise_fairness,
                )
            else:
                groupwise_fairness = np.where(
                    np.isnan(groupwise_fairness), 0, groupwise_fairness
                )

            self.step_fairness.append(groupwise_fairness)

            # Compute the weights
            self.cumulative_fairness = np.clip(
                self.cumulative_fairness
                + self.fairness_rate
                * np.hstack(
                    (
                        groupwise_fairness - self.epsilon,
                        -(groupwise_fairness + self.epsilon),
                    )
                ),
                0,
                None,
            )

            lag_parameters = (
                self.cumulative_fairness[:, : self.s_unique.shape[0]]
                - self.cumulative_fairness[:, self.s_unique.shape[0] :]
            )
            partial_weights = (
                self.fairness_function.C.T.dot(lag_parameters.reshape(-1, 1))
            ).reshape(self.y_unique.shape[0], self.s_unique.shape[0])
            weights = 1 + partial_weights / self.fairness_function.P
            self.step_loss.append(loss)
            self.step_weights.append(weights)
            device = loss.get_device()
            if device >= 0:
                device = f"cuda:{device}"
            else:
                device = torch.device("cpu")
            weighted_loss = loss * torch.tensor(
                (weights)[target.cpu().detach().numpy(), s.cpu().detach().numpy()],
                device=device,
            )
            self.step_weighted_loss.append(weighted_loss.mean())
            if self.reduction == "mean":
                return torch.mean(weighted_loss)
            elif self.reduction == "sum":
                return torch.sum(weighted_loss)
            else:
                return weighted_loss
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
