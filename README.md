# FairGrad: Fairness Aware Gradient Descent

FairGrad, is an easy to use general purpose approach to enforce fairness for gradient descent based methods. 

# Getting started: 
You can get ```fairgrad``` from pypi, which means it can be easily installed via ```pip```:
```
pip install fairgrad
```

# Example usage 
To use fairgrad simply replace your pytorch cross entropy loss with fairgrad cross entropy loss. 
Alongside, regular pytorch cross entropy arguments, it expects following meta data in the form of python dictionary.

```
fairness_related_arguments = {
  'y_train' : All train example's corresponding label (np.asarray[int]).
  's_train' : All train example's corresponding sensitive attribute. This means if there
            are 2 sensitive attributes, with each of them being binary. For instance gender - (male and female) and
            age (above 45, below 45). Total unique sentive attributes are 4 (np.asarray[int]).
  'fairness_measure': Currently we support "equal_odds", "equal_opportunity", and "accuracy_parity" (string). 
  'epsilon': The slack which is allowed for the final fairness level (float). 
  'fairness_rate': Parameter which intertwines current fairness weights with sum of previous fairness rates.
}
```

```python
# Note this is short snippet. One still needs to models and iterators.
# Full worked out example is available here - @TODO

from fairgrad.cross_entropy import CrossEntropyLoss

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

# Citation
```
@article{maheshwari2022fairgrad,
  title={FairGrad: Fairness Aware Gradient Descent},
  author={Maheshwari, Gaurav and Perrot, Micha{\"e}l},
  journal={arXiv preprint arXiv:2206.10923},
  year={2022}
}
```
