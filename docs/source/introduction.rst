Introduction
------------

FairGrad, is an easy to use general purpose approach to enforce
fairness for gradient descent based methods. The core idea is to enforce fairness by iteratively learns
group specific weights based on whether they are advantaged or not. FairGrad is:

    * Simple and easy to integrate with no significant overhead
    * Supports Multiclass problems and can work with any gradient based methods
    * Supports various group fairness notions including exact and approximate fairness
    * Is competitive on various tasks including complex NLP and CV ones


Installation
------------

You can install FairGrad from PyPI with `pip` or your favorite package manager::

    pip install fairgrad


Quick Start
------------

To use fairgrad simply replace your pytorch cross entropy loss with
fairgrad cross entropy loss. Alongside, regular pytorch cross entropy arguments,
it expects following *extra* arguments::

    y_train (np.asarray[int], Tensor, optional): All train example's corresponding label
    s_train (np.asarray[int], Tensor, optional): All train example's corresponding sensitive attribute. This means if there
            are 2 sensitive attributes, with each of them being binary. For instance gender - (male and female) and
            age (above 45, below 45). Total unique sentive attributes are 4.
    fairness_measure (string): Currently we support "equal_odds", "equal_opportunity", and "accuracy_parity".
    epsilon (float, optional): The slack which is allowed for the final fairness level.
    fairness_rate (float, optional): Parameter which intertwines current fairness weights with sum of previous fairness rates.

Usage Example
-------------

Below is a simple example::

        >>> from fairgrad.torch import CrossEntropyLoss
        >>> input = torch.randn(10, 5, requires_grad=True)
        >>> target = torch.empty(10, dtype=torch.long).random_(2)
        >>> s = torch.empty(10, dtype=torch.long).random_(2) # protected attribute
        >>> loss = CrossEntropyLoss(y_train = target, s_train = s, fairness_measure = 'equal_odds')
        >>> output = loss(input, target, s, mode='train')
        >>> output.backward()
