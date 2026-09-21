"""Run GMM clustering experiments for K=1 through K=5."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .model import GaussianMixture


def main():
    fig, ax = plt.subplots(1, 5, figsize=(24, 4))

    data_path = Path(__file__).resolve().parents[1] / "FAA_AEDT_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "FAA_AEDT_data.csv is not included in the repository. "
            "Place the dataset in the project root to reproduce the "
            "original coursework experiment. The model tests do not "
            "require this dataset."
        )

    data = pd.read_csv(data_path)

    for k in range(5):
        gmm = GaussianMixture(n_components=k + 1)
        gmm.fit(data)
        labels = gmm.predict(data)
        bic = gmm.BIC(data)

        ax[k].scatter(data["x1"], data["x2"], s=2, c=labels)
        ax[k].set_title(f"GMM with {k + 1} components")
        ax[k].set_xlabel(f"BIC = {bic:.2f}")

    plt.show()


if __name__ == "__main__":
    main()
