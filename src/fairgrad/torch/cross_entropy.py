import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
import numpy.typing as npt
from fairgrad.fairness_functions import *
from typing import Callable, Optional, Union, Type
from fairgrad.torch.fairness_loss import FairnessLoss


class CrossEntropyLoss(FairnessLoss):
    r"""This is an extension of the CrossEntropyLoss provided by pytorch. Please check pytorch documentation
    for understanding the cross entropy loss.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        fairness_measure (string, FairnessMeasure): Currently supported are "equal_odds", "equal_opportunity", "demographic_parity", and "accuracy_parity".
        y_train (np.asarray[int], Tensor, optional): All train example's corresponding label
        s_train (np.asarray[int], Tensor, optional): All train example's corresponding sensitive attribute. This means if there
            are 2 sensitive attributes, with each of them being binary. For instance gender - (male and female) and
            age (above 45, below 45). Total unique sentive attributes are 4.
        y_desirable (np.asarray[int], Tensor, optional): All desirable labels, only used with equality of opportunity.
        epsilon (float, optional): The slack which is allowed for the final fairness level.
        fairness_rate (float, optional): Parameter which intertwines current fairness weights with sum of previous fairness rates.
        **kwargs: Arbitrary keyword arguments passed to CrossEntropyLoss upon instantiation. Using is at your own risk as it might result in unexpected behaviours.

    Examples::

        >>> input = torch.randn(10, 5, requires_grad=True)
        >>> target = torch.empty(10, dtype=torch.long).random_(2)
        >>> s = torch.empty(10, dtype=torch.long).random_(2) # protected attribute
        >>> loss = CrossEntropyLoss(y_train = target, s_train = s, fairness_measure = 'equal_odds')
        >>> output = loss(input, target, s, mode='train')
        >>> output.backward()
    """

    def __init__(
        self,
        reduction: str = "mean",
        fairness_measure: Optional[Union[str, Type[FairnessMeasure]]] = None,
        y_train: Optional[Union[npt.NDArray[int], Tensor]] = None,
        s_train: Optional[Union[npt.NDArray[int], Tensor]] = None,
        y_desirable: Optional[Union[npt.NDArray[int], Tensor]] = [1],
        epsilon: Optional[float] = 0.0,
        fairness_rate: Optional[float] = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(
            nn.CrossEntropyLoss,
            reduction=reduction,
            fairness_measure=fairness_measure,
            y_train=y_train,
            s_train=s_train,
            y_desirable=y_desirable,
            epsilon=epsilon,
            fairness_rate=fairness_rate,
            **kwargs,
        )
