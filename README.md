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

```python
# Note this is short snippet. One still needs to models and iterators.
# Full worked out example is available here - @TODO

from fairgrad.torch import CrossEntropyLoss

# define cross entropy loss 
criterion = CrossEntropyLoss(fairness_related_meta_data=fairness_related_meta_data)

# Train loop

for inputs, labels, protected_attributes in train_iterator:
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, labels, protected_attributes, mode='train')
    loss.backward()
    optimizer.step()
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
