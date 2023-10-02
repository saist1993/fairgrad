# FairGrad: Fairness Aware Gradient Descent
[![Documentation Status](https://readthedocs.org/projects/fairgrad/badge/?version=latest)](https://fairgrad.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/fairgrad.svg)](https://badge.fury.io/py/fairgrad)
[![GitHub Actions (Tests)](https://github.com/saist1993/fairgrad/actions/workflows/build.yaml/badge.svg)](https://github.com/saist1993/fairgrad/actions/workflows/build.yaml)
<a href="https://arxiv.org/abs/2206.10923"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>

FairGrad, is an easy to use general purpose approach to enforce fairness for gradient descent based methods. 

# Getting started: 
You can get ```fairgrad``` from pypi, which means it can be easily installed via ```pip```:
```
pip install fairgrad
```

# Documentation
The documenation can be found at [read the docs](https://fairgrad.readthedocs.io/en/latest/index.html)

# Example usage 
To use fairgrad simply replace your pytorch cross entropy loss with fairgrad cross entropy loss. 
Alongside, regular pytorch cross entropy arguments, it expects following extra arguments.

```
y_train (np.asarray[int], Tensor, optional): All train example's corresponding label
s_train (np.asarray[int], Tensor, optional): All train example's corresponding sensitive attribute. This means if there
        are 2 sensitive attributes, with each of them being binary. For instance gender - (male and female) and
        age (above 45, below 45). Total unique sentive attributes are 4.
fairness_measure (string): Currently we support "equal_odds", "equal_opportunity", "accuracy_parity", and 
                           "demographic_parity". Note that demographic parity is only supported for binary case.
epsilon (float, optional): The slack which is allowed for the final fairness level.
fairness_rate (float, optional): Parameter which intertwines current fairness weights with sum of previous fairness rates.
```

Below is a small example snippet. A fully worked out example is available [here](https://github.com/saist1993/fairgrad/blob/main/examples/simple_classification_dataset.py)
```python
import torch
from fairgrad.torch import CrossEntropyLoss
input = torch.randn(10, 5, requires_grad=True)
target = torch.empty(10, dtype=torch.long).random_(2)
s = torch.empty(10, dtype=torch.long).random_(2) # protected attribute
loss = CrossEntropyLoss(y_train = target, s_train = s, fairness_measure = 'equal_odds')
output = loss(input, target, s, mode='train')
output.backward()
```

We highly recommend to **standardize features** by removing the mean and scaling to unit variance.
This can be done using standard scalar module in sklearn.

# Citation
```
@article{maheshwari2022fairgrad,
  title={FairGrad: Fairness Aware Gradient Descent},
  author={Maheshwari, Gaurav and Perrot, Micha{\"e}l},
  journal={arXiv preprint arXiv:2206.10923},
  year={2022}
}
```
