import torch
import numpy as np
import torch.nn as nn
import numpy.typing as npt
from torch.optim import SGD
from tqdm.auto import tqdm
from utils import get_celeb_data
from fairgrad.torch import CrossEntropyLoss
from sklearn.model_selection import train_test_split


def train(
    X: npt.NDArray,
    y: npt.NDArray[int],
    s: npt.NDArray[int],
    model: torch.nn,
    optimizer: torch.optim,
    criterion: CrossEntropyLoss,
    batch_size: int = 64,
    n_iterations: int = 1000,
):
    """
    Does the actual training of the model. It is just a typical pytorch training loop.
    :param X: Input features.
    :param y: Gold labels.
    :param s: Protected Attribute labels.
    :param model: Pytorch model.
    :param optimizer: Pytorch optimizer like SGD.
    :param criterion: Custom fairgrad loss function.
    :param batch_size: batch size.
    :param n_iterations: The number of iteration for training.
    :return: None.
    """

    for _ in tqdm(range(n_iterations)):  # iterator
        if batch_size is None:  # find a batch size element.
            mask = np.arange(X.shape[0])
        else:
            mask = np.random.choice(X.shape[0], size=batch_size, replace=False)
        model.train()
        optimizer.zero_grad()
        output = model(torch.tensor(X[mask, :]).float())
        loss = criterion(
            output, torch.tensor(y[mask]), torch.tensor(s[mask]), mode="train"
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()


def evaluate(
    X: npt.NDArray,
    y: npt.NDArray[int],
    s: npt.NDArray[int],
    model: torch.nn,
    criterion: CrossEntropyLoss,
):
    """

    :param X: Input features.
    :param y: Gold labels.
    :param s: Protected Attribute labels.
    :param model: Pytorch model.
    :param criterion: Custom fairgrad loss function.
    :return: accuracy, group_fairness_matrix and mean group fairness
    """
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X).float())
    groupwise_fairness = criterion.fairness_function.groupwise(
        output.detach().numpy(), y, s
    )
    accuracy = np.mean(y == np.argmax(output.detach().numpy(), axis=1).ravel())
    return accuracy, groupwise_fairness, np.mean(np.abs(groupwise_fairness))


def main():

    X, y_ori, s_ori = get_celeb_data()  # generates the data

    # Train, valid, test splits
    X_t, X_test, y_t, y_test, s_t, s_test = train_test_split(
        X, y_ori, s_ori, test_size=0.2
    )
    X_train, X_valid, y_train, y_valid, s_train, s_valid = train_test_split(
        X_t, y_t, s_t, test_size=0.25
    )

    # Setting up the problem
    fairness_function = "equal_odds"

    # Setting up the model and optimizer
    model = nn.Linear(X.shape[1], np.unique(y_train).shape[0])
    learning_rate = 1
    batch_size = 2048
    n_iterations = 2500
    optimizer = SGD(model.parameters(), lr=learning_rate)

    criterion = CrossEntropyLoss(
        y_train=y_train,
        s_train=s_train,
        fairness_measure=fairness_function,
        n_iterations=n_iterations,
    )

    print("training the model")
    train(
        X=X_train,
        y=y_train,
        s=s_train,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        n_iterations=1000,
    )

    print("training metrics")
    accuracy, groupwise_fairness, mean_group_fairness = evaluate(
        X=X_train, y=y_train, s=s_train, model=model, criterion=criterion
    )
    print(f"accuracy: {accuracy}")
    print(f"groupwise_fairness: {groupwise_fairness}")
    print(f"mean_group_fairness: {mean_group_fairness}")

    print("test metrics")
    accuracy, groupwise_fairness, mean_group_fairness = evaluate(
        X=X_train, y=y_train, s=s_train, model=model, criterion=criterion
    )
    print(f"accuracy: {accuracy}")
    print(f"groupwise_fairness: {groupwise_fairness}")
    print(f"mean_group_fairness: {mean_group_fairness}")


if __name__ == "__main__":
    main()
