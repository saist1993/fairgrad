import warnings

from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

import torch
import numpy as np
from torch import Tensor
from .fairness_functions import *
from typing import Callable, Optional


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
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch. If the
    :attr:`weight` argument is specified then this is a weighted average:

    .. math::
        \text{loss} = \frac{\sum^{N}_{i=1} loss(i, class[i])}{\sum^{N}_{i=1} weight[class[i]]}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


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
        fairness_related_meta_data (dict, optional): Specifies all fairness related meta data which are
            necessary for implementing fairgrad. This includes following information
            y_train (np.asarray[int]): All train example's corresponding label
            s_train (np.asarray[int]): All train example's corresponding sensitive attribute. This means if there
            are 2 sensitive attributes, with each of them being binary. For instance gender - (male and female) and
            age (above 45, below 45). Total unique sentive attributes are 4.
            fairness_measure (string): Currently we support "equal_odds", "equal_opportunity", and "accuracy_parity".
            epsilon (float): The slack which is allowed for the final fairness level.
            fairness_rate (float): Parameter which intertwines current fairness weights with sum of previous fairness rates.


    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
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
        fairness_related_meta_data: dict = {},
    ) -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        if fairness_related_meta_data:
            # Below keys is necessary
            assert "y_train" in fairness_related_meta_data.keys()
            assert "s_train" in fairness_related_meta_data.keys()
            assert "fairness_measure" in fairness_related_meta_data.keys()

            # Following are optional. fairness_rate of 0.01 seems to work well for wide range of task.
            try:
                self.epsilon = fairness_related_meta_data["epsilon"]
            except KeyError:
                self.epsilon = 0.0
            try:
                self.fairness_rate = fairness_related_meta_data["fairness_rate"]
            except KeyError:
                self.fairness_rate = 0.01

            self.y_train = fairness_related_meta_data["y_train"]
            self.s_train = fairness_related_meta_data["s_train"]
            if type(self.y_train) == torch.Tensor:
                self.y_train = fairness_related_meta_data["y_train"].numpy()
            if type(self.s_train) == torch.Tensor:
                self.s_train = fairness_related_meta_data["s_train"].numpy()

            self.y_unique = np.unique(self.y_train)
            self.s_unique = np.unique(self.s_train)
            self.fairness_measure = fairness_related_meta_data["fairness_measure"]
            self.fairness_function = get_fairness_function(
                self.fairness_measure, self.y_train, self.s_train
            )
            self.fairness_flag = True

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

        else:
            self.fairness_flag = False

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
                input.detach().numpy(), target.detach().numpy(), s.detach().numpy()
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
            weighted_loss = loss * torch.tensor((weights)[target, s])
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
