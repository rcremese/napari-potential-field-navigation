import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Plot loss functions in a csv file
def plot_losses(csv_filepath: Path):
    df = pd.read_csv(csv_filepath)
    assert df.columns.tolist() == ["total", "distance", "bending", "obstacle"]
    xticks = np.arange(0, len(df), 1)
    plt.figure()
    plt.plot(df["total"], color="blue")
    plt.xlabel("iter")
    plt.xticks(xticks)
    plt.ylabel("total loss")
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(df["distance"], color="red")
    ax[0].set(xlabel="iter", ylabel="L2 distance loss")
    ax[1].plot(df["bending"], color="green")
    ax[1].set(xlabel="iter", ylabel="bending loss")
    ax[2].plot(df["obstacle"], color="blue")
    ax[2].set(
        xlabel="iter", ylabel="obstacle loss", xticks=np.arange(0, len(df), 1)
    )
    plt.show()


if __name__ == "__main__":
    plot_losses(Path("C://Users//rcremese//Downloads//test_loss.csv"))
