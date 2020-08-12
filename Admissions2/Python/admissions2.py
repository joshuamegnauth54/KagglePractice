"""Playing around with an admissions data set from Kaggle.

Data from: https://www.kaggle.com/mohansacharya/graduate-admissions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve

# Dracula colors
# https://draculatheme.com/
# Obviously not following the theme yet.
PINK = "#ff79c6"
CYAN = "#8be9fd"
GREEN = "#50fa7b"
PURPLE = "#bd93f9"
RED = "#ff5555"
BACKGROUND = "#282a36"

# Various globals
HIST_KWARGS = {"edgecolor": BACKGROUND, "linewidth": 2}
FIGSIZE = (18, 16)
THRESHOLD = 0.65
SEED = 314


def set_options():
    """Set global options.

    (I clearly barely used this.)
    """
    sns.set_context("poster")


def clean_admissions(path: str) -> pd.DataFrame:
    """Load and clean admissions data.

    Parameters
    ----------
    path : str
        Path to admissions data.

    Returns
    -------
    Data in a Pandas DataFrame.

    """
    if not path:
        raise ValueError

    admissions = pd.read_csv(path,
                             header=0,
                             names=["id", "gre", "toefl", "uni_ratings",
                                    "statement", "letter", "cgpa",
                                    "research", "prob_admit"],
                             low_memory=False,
                             dtype={"research": bool,
                                    "uni_ratings": "category"})

    admissions["y_admit"] = admissions.prob_admit > THRESHOLD

    return admissions


def standardize(variable: pd.Series) -> pd.Series:
    """Standardize a variable by substracting the mean from each value and
    dividing by the standard deviation.

    Parameters
    ----------
    variable : pd.Series
        The variable to standardize.

    Raises
    ------
    ValueError
        The Series passed in must have at least one row.

    Returns
    -------
    pandas.Series
        Returns the standardized variable.

    """
    if not len(variable):
        raise ValueError("Series needs to be >0")

    return (variable - variable.mean())/variable.std()


def eda_tests_plots(admissions: pd.DataFrame, size=18):
    """Generates some not so fancy EDA plots."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=FIGSIZE)

    fig.suptitle("GRE, TOEFL, and GPA", fontsize=size, weight="bold")

    # Standardize GRE and TOEFL
    gre_stdize = standardize(admissions.gre)
    toefl_stdize = standardize(admissions.toefl)

    # Histograms of standardized GRE and TOEFL
    sns.distplot(gre_stdize, norm_hist=True, color=PINK,
                 hist_kws=HIST_KWARGS, ax=axes[0, 0])
    sns.distplot(toefl_stdize, norm_hist=True, color=GREEN,
                 axlabel="GRE [pink]/TOEFL [green]",
                 hist_kws=HIST_KWARGS, ax=axes[0, 0])

    # Scatter plot of standardized GRE and TOEFL scores
    sns.scatterplot(gre_stdize, toefl_stdize, color=PINK, ax=axes[0, 1])
    axes[0, 1].set_xlabel("Standardized GRE")
    axes[0, 1].set_ylabel("Standardized TOEFL")

    # Scatter plot of GRE versus CGPA
    sns.scatterplot("cgpa", "gre", "uni_ratings", data=admissions,
                    ax=axes[1, 0])
    axes[1, 0].set_xlabel("GPA")
    axes[1, 0].set_ylabel("GRE")

    # Scatter plot of TOEFL versus GPA
    sns.scatterplot("cgpa", "toefl", "prob_admit", data=admissions,
                    ax=axes[1, 1])
    axes[1, 1].set_xlabel("GPA")
    axes[1, 1].set_ylabel("TOEFL")

    return (fig, axes)


def admissions_fe(admissions: pd.DataFrame):
    """Creates tests and extra_docs variables via feature engineering."""
    admissions["tests"] = admissions.gre + admissions.toefl
    admissions["extra_docs"] = admissions.statement + admissions.letter

    admissions.tests = standardize(admissions.tests)
    admissions.extra_docs = standardize(admissions.extra_docs)
    admissions.cgpa = standardize(admissions.cgpa)
    admissions.drop(columns=["id", "gre", "toefl", "statement", "letter",
                             "prob_admit"], inplace=True)

    return pd.get_dummies(admissions, drop_first=True)


def admissions_split(admissions: pd.DataFrame):
    """Split the admissions DataFrame via train_test_split."""
    X = admissions.drop(columns=["y_admit"])
    y = admissions.y_admit.values

    return train_test_split(X, y, test_size=0.1, random_state=SEED)


def admissions_rf_model(X_train, y_train):
    """Fit and run (something of) a Python port of the model I built in R."""
    rf_grid = {"n_estimators": [32, 128, 256],
               "criterion": ["gini", "entropy"],
               "max_depth": np.concatenate((np.arange(20, 60, step=20),
                                           np.arange(1, 11, step=1)),
                                           axis=None),
               "max_features": list(range(1, len(X_train.columns))),
               "min_samples_split": np.linspace(0.05, 1.0, 10)
               }

    gridcv = GridSearchCV(RandomForestClassifier(random_state=SEED),
                          param_grid=rf_grid,
                          scoring=make_scorer(roc_auc_score),
                          n_jobs=-1,
                          cv=3)

    return gridcv.fit(X_train, y_train)


def plot_roc(model, X_test, y_test):
    """Plot the ROC curve with the AUC."""
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, label="AUC: {}".format(auc_score))
    ax.legend(loc=4)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC curve [feature engineered model]")

    return fig, ax


def main():
    """Runs all of the above in the lamely propietary manner in which I
    designed my code.

    I can't think of a good way to structure Python data science scripts as
    opposed to programs written in Python.
    """
    set_options()

    # Get the path to admissions2 from our command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the Admissions2 data set.")
    args = parser.parse_args()
    path = args.path

    # Plots
    admissions = clean_admissions(path)
    eda_tests_plots(admissions)

    # Final model as per the Rmd
    admissions = admissions_fe(admissions)
    admissions.head()
    X_train, X_test, y_train, y_test = admissions_split(admissions)
    mod = admissions_rf_model(X_train, y_train)

    # Plot the ROC curve with the AUC
    fig, ax = plot_roc(mod, X_test, y_test)
    fig.show()


if __name__ == "__main__":
    main()
