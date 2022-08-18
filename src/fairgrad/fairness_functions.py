import numpy as np
from typing import NamedTuple, Optional


def convert(preds):
    if len(preds.shape) == 1:
        return preds
    return np.argmax(preds, axis=1).ravel()


class FairnessMeasure:
    def __init__(self, y_unique, s_unique, y, s):
        self.y_unique = y_unique
        self.s_unique = s_unique
        self.init_C(y, s)
        self.init_P(y, s)


class EqualizedOdds(FairnessMeasure):
    def init_C(self, y, s):
        n_groups = self.y_unique.shape[0] * self.s_unique.shape[0]
        self.C = np.zeros((n_groups, n_groups))

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
                    self.C[i, j] = np.sum(s[y == lp] == rp) / np.sum(y == lp)
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
    def init_C(self, y, s):
        n_groups = self.y_unique.shape[0] * self.s_unique.shape[0]
        self.C = np.zeros((n_groups, n_groups))
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
    def __init__(self, y_desirable, y_unique, s_unique):
        self.y_desirable = y_desirable
        self.y_unique = y_unique
        self.s_unique = s_unique

    def groupwise(self, preds, y, s):
        preds = convert(preds)

        groupwise_fairness = np.zeros((self.y_unique.shape[0], self.s_unique.shape[0]))

        for l in self.y_desirable:
            reference_rate = np.mean(preds[y == l] == l)
            for r in self.s_unique:
                mask = np.logical_and(y == l, s == r)

                groupwise_fairness[l, r] = np.mean(preds[mask] == l) - reference_rate

        return groupwise_fairness


class FairnessSetupArguments(NamedTuple):
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
        )
    elif fairness_setup.fairness_function_name == "accuracy_parity":
        fairness_function = AccuracyParity(
            y_unique=fairness_setup.y_unique,
            s_unique=fairness_setup.s_unique,
            y=fairness_setup.all_train_y,
            s=fairness_setup.all_train_s,
        )
    else:
        raise NotImplementedError
    return fairness_function
