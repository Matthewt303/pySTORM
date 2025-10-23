import numpy as np
import matplotlib.pyplot as plt
from pySTORM.internals.image_analysis import dog_filter, extract_local_maxima


def filt_fit(image: "np.ndarray", threshold: float) -> "np.ndarray":
    filt_im = dog_filter(image)

    local_maxima = extract_local_maxima(filt_im)

    return local_maxima


def show_image(image: "np.ndarray", maxima: "np.ndarray") -> None:
    fig, ax = plt.subplots()

    ax.imshow(image, cmap="gray")
    ax.scatter(
        maxima[:, 1],
        maxima[:, 0],
        s=100,
        marker="s",
        facecolor="none",
        edgecolor="r",
        alpha=0.5,
        linewidths=2.0,
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", width=0)

    plt.show()
