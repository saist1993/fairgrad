import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import NamedTuple, Optional


def convert(preds):
    if len(preds.shape) == 1:
        return preds
    return np.argmax(preds, axis=1).ravel()


class FairnessMeasure:
    def __init__(
        self,
        y_unique: npt.NDArray[int],
        s_unique: npt.NDArray[int],
        y: npt.NDArray[int],
        s: npt.NDArray[int],
    ):
        self.y_unique = y_unique
        self.s_unique = s_unique
        self.init_C(y, s)
        self.init_P(y, s)


class EqualizedOdds(FairnessMeasure):
    r"""The function implements the equal odds fairness function. A model :math:`h_θ` is fair for Equalized Odds when
    the probability of predicting the correct label is independent of the sensitive attribute.

    Args:
        y_unique (npt.ndarray[int]): all unique labels in all label space.
        s_unique (npt.ndarray[int]): all unique protected attributes in all protected attribute space.
        y (npt.ndarray[int]): all label space
        s (npt.ndarray[int]): all protected attribute space
    """

    def init_C(self, y, s):
        n_groups = self.y_unique.shape[0] * self.s_unique.shape[0]
        self.C = np.zeros((n_groups, n_groups))
        self.C0 = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in self.y_unique for r in self.s_unique],
                [r for l in self.y_unique for r in self.s_unique],
            ),
            (self.y_unique.shape[0], self.s_unique.shape[0]),
        )

        for i in indices:
            for j in indices:
                l, r = np.unravel_index(
                    i, (self.y_unique.shape[0], self.s_unique.shape[0])
                )
                lp, rp = np.unravel_index(
                    j, (self.y_unique.shape[0], self.s_unique.shape[0])
                )
                if l == lp:
                    self.C[i, j] = np.mean(s[y == lp] == rp)
        self.C = self.C - np.eye(n_groups)

    def init_P(self, y, s):
        self.P = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))
        for l in self.y_unique:
            for r in self.s_unique:
                self.P[l, r] = np.mean(np.logical_and(y == l, s == r))

    def groupwise(self, preds, y, s):
        preds = convert(preds)

        groupwise_fairness = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))

        for l in self.y_unique:
            reference_rate = np.mean(preds[y == l] == l)
            for r in self.s_unique:
                mask = np.logical_and(y == l, s == r)
                groupwise_fairness[l, r] = np.mean(preds[mask] == l) - reference_rate

        return groupwise_fairness


class AccuracyParity(FairnessMeasure):
    r"""The function implements the accuracy parity fairness function.
    A model :math:`h_θ` is fair for Accuracy Parity when theprobability of being correct is independent of the sensitive attribute.

    Args:
        y_unique (npt.ndarray[int]): all unique labels in all label space.
        s_unique (npt.ndarray[int]): all unique protected attributes in all protected attribute space.
        y (npt.ndarray[int]): all label space
        s (npt.ndarray[int]): all protected attribute space
    """

    def init_C(self, y, s):
        n_groups = self.y_unique.shape[0] * self.s_unique.shape[0]
        self.C = np.zeros((n_groups, n_groups))
        self.C0 = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in self.y_unique for r in self.s_unique],
                [r for l in self.y_unique for r in self.s_unique],
            ),
            (self.y_unique.shape[0], self.s_unique.shape[0]),
        )

        for j in indices:
            lp, rp = np.unravel_index(
                j, (self.y_unique.shape[0], self.s_unique.shape[0])
            )
            self.C[:, j] = np.mean(s == rp)

        self.C = (
            self.C
            - np.eye(n_groups)
            - np.eye(n_groups, k=self.s_unique.shape[0])
            - np.eye(n_groups, k=-self.s_unique.shape[0])
        ) / self.y_unique.shape[0]

    def init_P(self, y, s):
        self.P = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))
        for r in self.s_unique:
            self.P[:, r] = np.mean(s == r)

    def groupwise(self, preds, y, s):
        preds = convert(preds)

        groupwise_fairness = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))

        reference_rate = np.mean(preds == y)
        for r in self.s_unique:
            mask = s == r
            groupwise_fairness[:, r] = np.mean(preds[mask] == y[mask]) - reference_rate

        return groupwise_fairness


class EqualityOpportunity:
    r"""The function implements the accuracy parity fairness function.
    A model :math:`h_θ` is fair for Equality of Opportunity when the probability of predicting the
    correct label is independent of the sensitive attribute for a given subset of labels called the desirable outcomes

    Args:
        y_unique (npt.ndarray[int]): all unique labels in all label space.
        s_unique (npt.ndarray[int]): all unique protected attributes in all protected attribute space.
        y (npt.ndarray[int]): all label space
        s (npt.ndarray[int]): all protected attribute space
        y_desirable (npt.ndarray[int]): the label for which the fairness needs to be enforced.
    """

    def __init__(
        self,
        y_unique: npt.NDArray[int],
        s_unique: npt.NDArray[int],
        y: npt.NDArray[int],
        s: npt.NDArray[int],
        y_desirable: npt.NDArray[int],
    ):
        self.y_desirable = y_desirable
        self.y_unique = y_unique
        self.s_unique = s_unique
        self.y = y
        self.s = s
        self.init_C(self.y, self.s)
        self.init_P(self.y, self.s)

    def init_C(self, y, s):
        n_groups = self.y_unique.shape[0] * self.s_unique.shape[0]
        self.C = np.zeros((n_groups, n_groups))
        self.C0 = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in self.y_unique for r in self.s_unique],
                [r for l in self.y_unique for r in self.s_unique],
            ),
            (self.y_unique.shape[0], self.s_unique.shape[0]),
        )

        for i in indices:
            for j in indices:
                l, r = np.unravel_index(
                    i, (self.y_unique.shape[0], self.s_unique.shape[0])
                )
                lp, rp = np.unravel_index(
                    j, (self.y_unique.shape[0], self.s_unique.shape[0])
                )
                if l == lp:
                    self.C[i, j] = np.mean(s[y == lp] == rp)
        self.C = self.C - np.eye(n_groups)
        for i in indices:
            l, r = np.unravel_index(i, (self.y_unique.shape[0], self.s_unique.shape[0]))
            if l not in self.y_desirable:
                self.C[i, :] = 0

    def init_P(self, y, s):
        self.P = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))
        for l in self.y_unique:
            for r in self.s_unique:
                self.P[l, r] = np.mean(np.logical_and(y == l, s == r))

    def groupwise(self, preds, y, s):
        preds = convert(preds)

        groupwise_fairness = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))

        for l in self.y_desirable:
            reference_rate = np.mean(preds[y == l] == l)
            for r in self.s_unique:
                mask = np.logical_and(y == l, s == r)

                groupwise_fairness[l, r] = np.mean(preds[mask] == l) - reference_rate

        return groupwise_fairness


class DemographicParity(FairnessMeasure):
    r"""The function implements the demographic parity fairness function. A model :math:`h_θ` is fair for Demographic Parity when
    the probability of predicting each label is independent of the sensitive attribute.

    Args:
        y_unique (npt.ndarray[int]): all unique labels in all binary label space.
        s_unique (npt.ndarray[int]): all unique protected attributes in all protected attribute space.
        y (npt.ndarray[int]): all binary label space
        s (npt.ndarray[int]): all protected attribute space
    """

    def __init__(
        self,
        y_unique: npt.NDArray[int],
        s_unique: npt.NDArray[int],
        y: npt.NDArray[int],
        s: npt.NDArray[int],
    ):
        super().__init__(y_unique, s_unique, y, s)
        if y_unique.shape[0] != 2:
            raise ValueError(
                "Demographic Parity is only applicable to binary problems (y_unique contained {} classes.".format(
                    y_unique.shape[0]
                )
            )

    def init_C(self, y, s):
        n_groups = self.y_unique.shape[0] * self.s_unique.shape[0]
        self.C = np.zeros((n_groups, n_groups))
        self.C0 = np.zeros((n_groups,))

        indices = np.ravel_multi_index(
            (
                [l for l in self.y_unique for r in self.s_unique],
                [r for l in self.y_unique for r in self.s_unique],
            ),
            (self.y_unique.shape[0], self.s_unique.shape[0]),
        )

        for i in indices:
            l, r = np.unravel_index(i, (self.y_unique.shape[0], self.s_unique.shape[0]))
            for j in indices:
                lp, rp = np.unravel_index(
                    j, (self.y_unique.shape[0], self.s_unique.shape[0])
                )

                if l == lp:
                    self.C[i, j] = np.mean(np.logical_and(y == lp, s == rp))
                    if r == rp:
                        self.C[i, j] = self.C[i, j] - np.mean(y[s == rp] == lp)
                else:
                    self.C[i, j] = -np.mean(np.logical_and(y == lp, s == rp))
                    if r == rp:
                        self.C[i, j] = self.C[i, j] + np.mean(y[s == rp] == lp)
            self.C0[i] = np.mean(y != l) - np.mean(y[s == r] != l)

    def init_P(self, y, s):
        self.P = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))
        for l in self.y_unique:
            for r in self.s_unique:
                self.P[l, r] = np.mean(np.logical_and(y == l, s == r))

    def groupwise(self, preds, y, s):
        preds = convert(preds)

        groupwise_fairness = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))

        for l in self.y_unique:
            reference_rate = np.mean(preds == l)
            for r in self.s_unique:
                mask = s == r
                groupwise_fairness[l, r] = np.mean(preds[mask] == l) - reference_rate

        return groupwise_fairness


@dataclass
class FairnessSetupArguments:
    fairness_function_name: str
    y_unique: np.asarray
    s_unique: np.asarray
    all_train_y: np.asarray
    all_train_s: np.asarray
    y_desirable: Optional[np.asarray] = None


def setup_fairness_function(fairness_setup: FairnessSetupArguments):
    if fairness_setup.fairness_function_name == "equal_odds":
        fairness_function = EqualizedOdds(
            y_unique=fairness_setup.y_unique,
            s_unique=fairness_setup.s_unique,
            y=fairness_setup.all_train_y,
            s=fairness_setup.all_train_s,
        )
    elif fairness_setup.fairness_function_name == "equal_opportunity":
        fairness_function = EqualityOpportunity(
            y_unique=fairness_setup.y_unique,
            s_unique=fairness_setup.s_unique,
            y_desirable=fairness_setup.y_desirable,
            y=fairness_setup.all_train_y,
            s=fairness_setup.all_train_s,
        )
    elif fairness_setup.fairness_function_name == "accuracy_parity":
        fairness_function = AccuracyParity(
            y_unique=fairness_setup.y_unique,
            s_unique=fairness_setup.s_unique,
            y=fairness_setup.all_train_y,
            s=fairness_setup.all_train_s,
        )
    elif fairness_setup.fairness_function_name == "demographic_parity":
        fairness_function = DemographicParity(
            y_unique=fairness_setup.y_unique,
            s_unique=fairness_setup.s_unique,
            y=fairness_setup.all_train_y,
            s=fairness_setup.all_train_s,
        )
    else:
        raise NotImplementedError
    return fairness_function
