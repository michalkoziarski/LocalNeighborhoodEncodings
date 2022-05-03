from pathlib import Path

import smote_variants as sv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import metrics

RESULTS_PATH = Path(__file__).parents[0] / "results"
STATS_PATH = Path(__file__).parents[0] / "stats"
RANDOM_STATE = 42


def get_classifiers():
    classifiers = {
        "CART": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(random_state=RANDOM_STATE),
    }

    return classifiers


def get_criteria():
    criteria = {
        "AUC": metrics.auc,
        "BAC": metrics.bac,
        "G-mean": metrics.g_mean,
    }

    return criteria


def get_scoring_functions():
    scoring_functions = {
        "Precision": metrics.precision,
        "Recall": metrics.recall,
        "AUC": metrics.auc,
        "BAC": metrics.bac,
        "G-mean": metrics.g_mean,
    }

    return scoring_functions


def get_reference_resamplers():
    resamplers = {
        "SMOTE": sv.SMOTE,
        "pf-SMOTE": sv.polynom_fit_SMOTE,
        "Lee": sv.Lee,
        "SMOBD": sv.SMOBD,
        "G-SMOTE": sv.G_SMOTE,
        "LVQ-SMOTE": sv.LVQ_SMOTE,
        "A-SMOTE": sv.Assembled_SMOTE,
        "SMOTE-TL": sv.SMOTE_TomekLinks,
    }

    return resamplers
