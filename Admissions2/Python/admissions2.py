"""Playing around with an admissions data set from Kaggle.

Data from: https://www.kaggle.com/mohansacharya/graduate-admissions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Dracula colors
# https://draculatheme.com/
# Obviously not following the theme yet.
PINK = "#ff79c6"
CYAN = "#8be9fd"
GREEN = "#50fa7b"
PURPLE = "#bd93f9"
RED = "#ff5555"
BACKGROUND = "#282a36"

HIST_KWARGS = {"edgecolor": BACKGROUND, "linewidth": 2}
FIGSIZE = (18, 16)
THRESHOLD = 0.65

def set_options():
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

    admissions.drop(columns="id", inplace=True)
    admissions["y_admit"] = admissions.prob_admit > THRESHOLD

    return admissions


def standardize(variable: pd.Series) -> pd.Series:
    """Standardize a variable by substracting the mean from each value and
    dividing by the standard deviation
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
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = FIGSIZE)

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


def admissions_log_model(admissions: pd.DataFrame):
    """I intended to just repeat my code from R, but honestly I am supremely
    bored doing so. I'll just recreate most of it later.

    For example! My first R model didn't shuffle and split the data into
    training and test sets. Do I really need to repeat all of that?"""

    X = admissions[["gre", "uni_ratings", "cgpa", "statement", "letter",
                    "research"]]
    y = admissions.y_admit.values
    basic_mod = LogisticRegression(n_jobs=-1)
    basic_mod.fit(X, y)


def main():
    set_options()

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help = "Path to the Admissions2 data set.")
    args = parser.parse_args()
    path = args.path

    admissions = clean_admissions(path)

if __name__ == "__main__":
    main()
