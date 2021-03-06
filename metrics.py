from collections import Counter

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
)


def metric_decorator(metric_function):
    def metric_wrapper(ground_truth, predictions, minority_class=None):
        if minority_class is None:
            minority_class = Counter(ground_truth).most_common()[-1][0]

        return metric_function(ground_truth, predictions, minority_class)

    return metric_wrapper


@metric_decorator
def precision(ground_truth, predictions, minority_class=None):
    return precision_score(
        ground_truth, predictions, pos_label=minority_class, zero_division=0
    )


@metric_decorator
def recall(ground_truth, predictions, minority_class=None):
    return recall_score(
        ground_truth, predictions, pos_label=minority_class, zero_division=0
    )


@metric_decorator
def auc(ground_truth, predictions, minority_class=None):
    return average_precision_score(ground_truth, predictions, pos_label=minority_class)


def bac(ground_truth, predictions):
    return balanced_accuracy_score(ground_truth, predictions)


def g_mean(ground_truth, predictions):
    return geometric_mean_score(ground_truth, predictions)


@metric_decorator
def f_beta(ground_truth, predictions, minority_class=None):
    n_minority = Counter(ground_truth).most_common()[1][1]
    n_majority = Counter(ground_truth).most_common()[0][1]

    return fbeta_score(
        ground_truth,
        predictions,
        beta=(n_majority / n_minority),
        pos_label=minority_class,
        zero_division=0,
    )
