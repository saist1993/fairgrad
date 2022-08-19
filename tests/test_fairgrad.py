import torch
import numpy as np
from fairgrad import __version__
from fairgrad.torch import CrossEntropyLoss


def test_version():
    assert __version__ == "0.1.1"


def test_complete_cross_entropy_loss_with_tensors():
    input = torch.randn(10, 5, requires_grad=True)
    target = torch.empty(10, dtype=torch.long).random_(2)
    s = torch.empty(10, dtype=torch.long).random_(2)  # protected attribute
    loss = CrossEntropyLoss(y_train=target, s_train=s, fairness_measure="equal_odds")
    output = loss(input, target, s, mode="train")
    output.backward()


def test_complete_cross_entropy_loss_with_numpy():
    input = torch.randn(10, 5, requires_grad=True)
    target = np.random.randint(0, 2, 10)
    s = np.random.randint(0, 2, 10)  # protected attribute
    loss = CrossEntropyLoss(y_train=target, s_train=s, fairness_measure="equal_odds")
    output = loss(input, torch.tensor(target), torch.tensor(s), mode="train")
    output.backward()


def test_all_fairness_measure():
    fairness_measures = ["equal_odds", "equal_opportunity", "accuracy_parity"]
    # fairness_measures = ['equal_opportunity']

    for fairness_measure in fairness_measures:
        input = torch.randn(10, 5, requires_grad=True)
        target = np.random.randint(0, 2, 10)
        s = np.random.randint(0, 2, 10)  # protected attribute
        loss = CrossEntropyLoss(
            y_train=target, s_train=s, fairness_measure=fairness_measure
        )
        output = loss(input, torch.tensor(target), torch.tensor(s), mode="train")
        output.backward()
