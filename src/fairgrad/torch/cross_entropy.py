import warnings

import torch.nn as nn
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

import torch
import numpy as np
from torch import Tensor
import numpy.typing as npt
from fairgrad.fairness_functions import *
from typing import Callable, Optional, Union, Type

class FairnessLoss(nn.modules.loss._Loss):
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
    __constants__ = ["reduction"]
    ignore_index: int

    def __init__(
        self,
        base_loss_class: Type[nn.modules.loss._Loss],
        reduction: str = 'mean',
        fairness_measure: Optional[Union[str, Type[FairnessMeasure]]] = None,
        y_train: Optional[Union[npt.NDArray[int], Tensor]] = None,
        s_train: Optional[Union[npt.NDArray[int], Tensor]] = None,
        y_desirable: Optional[Union[npt.NDArray[int], Tensor]] = [1],
        epsilon: Optional[float] = 0.0,
        fairness_rate: Optional[float] = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(reduction=reduction)
        self.base_loss = base_loss_class(reduction='none', **kwargs)
        self.reduction = reduction

        if fairness_measure is None:
            self.fairness_function = None
            warnings.warn("FairGrad mechanism is not employed as fairness_measure is missing. Reverting to standard {}.".format(type(self.base_loss).__name__), RuntimeWarning)
        else:
            if type(y_train) == torch.Tensor:                
                y_train = y_train.cpu().detach().numpy()
            if type(y_desirable) == torch.Tensor:
                y_desirable = y_desirable.cpu().detach().numpy()
            if type(s_train) == torch.Tensor:
                s_train = s_train.cpu().detach().numpy()
                
            self.fairness_function = instantiate_fairness(fairness_measure, y_train = y_train, s_train = s_train, y_desirable = y_desirable)
            
            self.epsilon = epsilon
            self.fairness_rate = fairness_rate
           
            # some lists to store all information over time. This in the future would be given as an option so
            # as to save space. @TODO: describe them individually.
            self.step_groupwise_weights = []
            self.step_loss = []
            self.step_weighted_loss = []
            self.step_accuracy = []
            self.step_fairness = []
            self.cumulative_fairness = torch.zeros(
                (self.fairness_function.y_unique.shape[0], 2 * self.fairness_function.s_unique.shape[0])
            )

    def reduce(self,loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
    def forward(
        self, input: Tensor, target: Tensor, s: Tensor = None
    ) -> Tensor:
        loss = self.base_loss(input, target)

        if s is None:
            self.fairness_function = None
            warnings.warn("FairGrad mechanism is not employed as the sensitive attribute s was not provided during the forward pass. is missing. Reverting to standard {}.".format(type(self.base_loss).__name__), RuntimeWarning)

        if self.fairness_function is not None:
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
                self.cumulative_fairness[:, : self.fairness_function.s_unique.shape[0]]
                - self.cumulative_fairness[:, self.fairness_function.s_unique.shape[0] :]
            )
            partial_weights = (
                self.fairness_function.C.T.dot(lag_parameters.reshape(-1, 1))
            ).reshape(self.fairness_function.y_unique.shape[0], self.fairness_function.s_unique.shape[0])
            weights = 1 + partial_weights / self.fairness_function.P

            self.step_loss.append(self.reduce(loss))
            self.step_groupwise_weights.append(weights*self.fairness_function.P)
            
            device = loss.get_device()
            if device >= 0:
                device = f"cuda:{device}"
            else:
                device = torch.device("cpu")
            weighted_loss = loss * torch.tensor(
                (weights)[target.cpu().detach().numpy(), s.cpu().detach().numpy()],
                device=device,
            )
            
            loss = weighted_loss
            self.step_weighted_loss.append(self.reduce(loss))

        return self.reduce(loss)

class CrossEntropyLoss(FairnessLoss):
    def __init__(self, **kwargs):
        super().__init__(nn.CrossEntropyLoss, **kwargs)
