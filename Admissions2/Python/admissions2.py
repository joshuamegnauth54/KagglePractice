"""Playing around with an admissions data set from Kaggle.

Data from: https://www.kaggle.com/mohansacharya/graduate-admissions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

def eda_tests_density(admissions, size=18):
    fig, axes=plt.subplots(ncols=2)

    fig.suptitle("GRE & TOEFL density plots", fontsize=size, weight="bold")

    sns.distplot(admissions.gre, hist=True, color="blue",
                 axlabel="GRE", ax=axes[0])
    sns.distplot(admissions.toefl, hist=True, color="red",
                 axlabel="TOEFL", ax=axes[1])

    return (fig, axes)

