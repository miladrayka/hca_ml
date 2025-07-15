"""This module contains several plots, i.e., pKa,
molecular properties, and T-SNE."""

from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils


def plot_pka(file_paths: List, colors: List) -> None:
    """Plot pKa KDE diagram.

    Parameters
    ----------
    file_paths : List
        A list of file path to several .csv files.
    colors : List
        A list of three colors.
    """

    dataframes = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    names = ["II", "IX", "XII"]

    plt.figure(figsize=(10, 8))

    for i, df in enumerate(dataframes):
        sns.kdeplot(
            x=df["pK"],
            label=names[i],
            color=colors[i],
            bw_adjust=5,
            cut=0,
            fill=False,
            lw=3,
        )
    plt.grid(visible=True, ls="--", lw=0.75)
    plt.xlabel("$pK=-log_{10}(K_{i}/K_{d}/IC50)$")
    plt.ylabel("Density")
    plt.legend(title="Isoform", fancybox=True)
    plt.savefig("../Data/pk_kde_plot.png", dpi=300, pad_inches=0.1, bbox_inches="tight")
    plt.show()


def plot_property(file_paths: List, colors: List) -> None:
    """Plot properties KDE diagram.

    Parameters
    ----------
    file_paths : List
        A list of file path to several .csv files.
    colors : List
        A list of three colors.
    """
    dataframes = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    properties = [
        "mw",
        "n_lipinski_hba",
        "n_lipinski_hbd",
        "qed",
        "clogp",
        "n_rotatable_bonds",
    ]

    property_name_dict = {
        "mw": "Molecular Weight",
        "n_lipinski_hba": "Number of Hydrogen Bond Acceptors",
        "n_lipinski_hbd": "Number of Hydrogen Bond Donors",
        "qed": "QED",
        "clogp": "LogP",
        "n_rotatable_bonds": "Number of Rotatable Bonds",
    }

    fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(15, 15))

    names = ["II", "IX", "XII"]

    for i, p in enumerate(properties):
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        for j, df in enumerate(dataframes):
            # file_name = file_paths[j].split("/")[-1][:5]
            sns.kdeplot(
                x=df[p],
                label=names[j],
                color=colors[j],
                bw_adjust=5,
                cut=0,
                fill=False,
                ax=ax,
                lw=2.5,
            )
        ax.grid(visible=True, ls="--", lw=0.75)
        ax.set_xlabel(property_name_dict[p])
        ax.set_ylabel("")

    fig.supylabel("Density", x=0.04, y=0.54, fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    plt.legend(title="Isoform", fancybox=True)

    plt.savefig("../Data/molecular_property_kde_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_t_sne(file_paths: List, colors: List) -> None:
    """Plot T-SNE diagram.

    Parameters
    ----------
    file_paths : List
        A list of file path to several .csv files.
    colors : List
        A list of three colors.
    """
    reps_dict = {}
    names = ["II", "IX", "XII"]
    for i, file_path in enumerate(file_paths):
        reps = utils.t_sne(file_path)
        reps_dict[names[i]] = reps

    colors = {"II": colors[0], "IX": colors[1], "XII": colors[2]}
    alphas = {"II": 0.9, "IX": 0.5, "XII": 0.3}
    plt.figure(figsize=(10, 10))
    for key in reps_dict.keys():
        plt.scatter(
            reps_dict[key][:, 0],
            reps_dict[key][:, 1],
            s=1.5,
            facecolors="none",
            alpha=alphas[key],
            label=key,
            linewidths=1.7,
            edgecolors=colors[key],
        )
    plt.legend(title="Isoform")
    plt.xlabel("T-SNE-1")
    plt.ylabel("T-SNE-2")
    plt.grid(True, linestyle="--", alpha=0.5, lw=0.75)
    plt.savefig("../Data/chemical_space_plot.png", dpi=300)
    plt.show()
