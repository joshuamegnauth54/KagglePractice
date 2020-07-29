"""Playing around with an admissions data set from Kaggle.

Data from: https://www.kaggle.com/mohansacharya/graduate-admissions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
                                    "uni_ratings": "category",
                                    "statement": "category",
                                    "letter": "category"})

    return admissions

def eda_tests_plots(admissions, size=18):
    fig, axes=plt.subplots(nrows=2, ncols=2, figsize=FIGSIZE)

    fig.suptitle("GRE & TOEFL density plots", fontsize=size, weight="bold")

    gre_stdize = (admissions.gre - admissions.gre.mean())/admissions.gre.std()
    toefl_stdize = (admissions.toefl - admissions.toefl.mean())/admissions.toefl.std()

    sns.distplot(gre_stdize, norm_hist=True, color=PINK,
                 hist_kws=HIST_KWARGS, ax=axes[0, 0])
    sns.distplot(toefl_stdize, norm_hist=True, color=GREEN,
                 axlabel="GRE/TOEFL", hist_kws=HIST_KWARGS, ax=axes[0, 0])

    sns.scatterplot(gre_stdize, toefl_stdize, color=GREEN, ax=axes[0, 1])

    sns.scatterplot("cgpa", "gre", "uni_ratings", data=admissions,
                    ax=axes[1, 0])
    sns.scatterplot("cgpa", "toefl", "prob_admit", data=admissions,
                    ax=axes[1, 1])

    return (fig, axes)

