import unittest

import torch
import numpy as np

# fairgrad imports.
from fairgrad import __version__
from fairgrad.torch.cross_entropy import CrossEntropyLoss
from fairgrad.fairness_functions import *
    

class TestFairnessMeasures(unittest.TestCase):
    def test_instantiate_fairness(self):
        y_train = np.random.randint(5, size=(1000,))
        s_train = np.random.randint(3, size=(1000,))
        with self.assertRaises(NotImplementedError):
            instantiate_fairness("dummy_parity",y_train,s_train)
            
        with self.assertRaises(ValueError):
            instantiate_fairness(22,y_train,s_train)
            instantiate_fairness(EqualizedOdds,None,s_train)
            instantiate_fairness(EqualizedOdds,y_train,None)
            instantiate_fairness(EqualizedOdds,y_train+1,s_train)
            instantiate_fairness(EqualizedOdds,y_train,s_train+1)
            instantiate_fairness(EqualityOpportunity,y_train,s_train,None)
            instantiate_fairness(EqualityOpportunity,y_train,s_train,np.array([0,2,5]))
            
        
    
    def test_EqualizedOdds(self):
        y_train = np.random.randint(5, size=(1000,))
        y_preds = np.random.randint(5, size=(1000,))
        s_train = np.random.randint(3, size=(1000,))
        fm = instantiate_fairness(EqualizedOdds,y_train,s_train)
        groupwise = fm.groupwise(y_preds,y_train,s_train)
    
        n_groups = fm.y_unique.shape[0] * fm.s_unique.shape[0]
        error_rates = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in fm.y_unique for r in fm.s_unique],
                [r for l in fm.y_unique for r in fm.s_unique],
            ),
            (fm.y_unique.shape[0], fm.s_unique.shape[0]),
        )

        for i in indices:
            l, r = np.unravel_index(
                i, (fm.y_unique.shape[0], fm.s_unique.shape[0])
            )
            error_rates[i] = np.mean((y_train != y_preds)[np.logical_and(y_train == l,s_train == r)])
            
        groupwise_C = fm.C.dot(error_rates.reshape(-1,1)) + fm.C0.reshape(-1,1)
        groupwise_C = groupwise_C.reshape(fm.y_unique.shape[0],fm.s_unique.shape[0])

        self.assertTrue(np.allclose(groupwise,groupwise_C))
        self.assertTrue(np.allclose(groupwise_C,groupwise))

    def test_EqualiOpportunity(self):
        y_train = np.random.randint(5, size=(1000,))
        y_preds = np.random.randint(5, size=(1000,))
        s_train = np.random.randint(3, size=(1000,))
        y_desirable = np.array([0,2,3])
        fm = instantiate_fairness(EqualityOpportunity,y_train,s_train,y_desirable)
        groupwise = fm.groupwise(y_preds,y_train,s_train)

        n_groups = fm.y_unique.shape[0] * fm.s_unique.shape[0]
        error_rates = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in fm.y_unique for r in fm.s_unique],
                [r for l in fm.y_unique for r in fm.s_unique],
            ),
            (fm.y_unique.shape[0], fm.s_unique.shape[0]),
        )

        for i in indices:
            l, r = np.unravel_index(
                i, (fm.y_unique.shape[0], fm.s_unique.shape[0])
            )
            error_rates[i] = np.mean((y_train != y_preds)[np.logical_and(y_train == l,s_train == r)])
            
        groupwise_C = fm.C.dot(error_rates.reshape(-1,1)) + fm.C0.reshape(-1,1)
        groupwise_C = groupwise_C.reshape(fm.y_unique.shape[0],fm.s_unique.shape[0])

        self.assertTrue(np.allclose(groupwise,groupwise_C))
        self.assertTrue(np.allclose(groupwise_C,groupwise))

    def test_AccuracyParity(self):
        y_train = np.random.randint(5, size=(1000,))
        y_preds = np.random.randint(5, size=(1000,))
        s_train = np.random.randint(3, size=(1000,))
        fm = instantiate_fairness(AccuracyParity,y_train,s_train)
        groupwise = fm.groupwise(y_preds,y_train,s_train)
    
        n_groups = fm.y_unique.shape[0] * fm.s_unique.shape[0]
        error_rates = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in fm.y_unique for r in fm.s_unique],
                [r for l in fm.y_unique for r in fm.s_unique],
            ),
            (fm.y_unique.shape[0], fm.s_unique.shape[0]),
        )

        for i in indices:
            l, r = np.unravel_index(
                i, (fm.y_unique.shape[0], fm.s_unique.shape[0])
            )
            error_rates[i] = np.mean((y_train != y_preds)[s_train == r])

        groupwise_C = fm.C.dot(error_rates.reshape(-1,1)) + fm.C0.reshape(-1,1)
        groupwise_C = groupwise_C.reshape(fm.y_unique.shape[0],fm.s_unique.shape[0])
        
        self.assertTrue(np.allclose(groupwise,groupwise_C))
        self.assertTrue(np.allclose(groupwise_C,groupwise))

    def test_DemographicParity(self):
        y_train = np.random.randint(2, size=(1000,))
        y_preds = np.random.randint(2, size=(1000,))
        s_train = np.random.randint(3, size=(1000,))
        fm = instantiate_fairness(DemographicParity,y_train,s_train)
        groupwise = fm.groupwise(y_preds,y_train,s_train)
    
        n_groups = fm.y_unique.shape[0] * fm.s_unique.shape[0]
        error_rates = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in fm.y_unique for r in fm.s_unique],
                [r for l in fm.y_unique for r in fm.s_unique],
            ),
            (fm.y_unique.shape[0], fm.s_unique.shape[0]),
        )

        for i in indices:
            l, r = np.unravel_index(
                i, (fm.y_unique.shape[0], fm.s_unique.shape[0])
            )
            error_rates[i] = np.mean((y_train != y_preds)[np.logical_and(y_train == l,s_train == r)])
            
        groupwise_C = fm.C.dot(error_rates.reshape(-1,1)) + fm.C0.reshape(-1,1)
        groupwise_C = groupwise_C.reshape(fm.y_unique.shape[0],fm.s_unique.shape[0])

        self.assertTrue(np.allclose(groupwise,groupwise_C))
        self.assertTrue(np.allclose(groupwise_C,groupwise))
        
def test_version():
    assert __version__ == "0.1.7"


def test_complete_cross_entropy_loss_with_tensors():
    test_input = torch.randn(10, 5, requires_grad=True)
    target = torch.empty(10, dtype=torch.long).random_(2)
    s = torch.empty(10, dtype=torch.long).random_(2)  # protected attribute
    loss = CrossEntropyLoss(y_train=target, s_train=s, fairness_measure="equal_odds")
    output = loss(test_input, target, s, mode="train")
    output.backward()


def test_complete_cross_entropy_loss_with_numpy():
    test_input = torch.randn(10, 5, requires_grad=True)
    target = np.random.randint(0, 2, 10)
    s = np.random.randint(0, 2, 10)  # protected attribute
    loss = CrossEntropyLoss(y_train=target, s_train=s, fairness_measure="equal_odds")
    output = loss(test_input, torch.tensor(target), torch.tensor(s), mode="train")
    output.backward()


def test_all_fairness_measure():
    fairness_measures = [
        "equal_odds",
        "equal_opportunity",
        "accuracy_parity",
        "demographic_parity",
    ]

    for fairness_measure in fairness_measures:
        test_input = torch.randn(10, 5, requires_grad=True)
        target = np.random.randint(0, 2, 10)
        s = np.random.randint(0, 2, 10)  # protected attribute
        loss = CrossEntropyLoss(
            y_train=target, s_train=s, fairness_measure=fairness_measure
        )
        output = loss(test_input, torch.tensor(target), torch.tensor(s), mode="train")
        output.backward()
