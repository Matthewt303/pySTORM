import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import sys
from pySTORM.internals.image_analysis import dog_filter, extract_local_maxima


def filt_fit(image: "np.ndarray", threshold: float) -> "np.ndarray":
    filt_im = dog_filter(image)

    rms = np.sqrt(np.mean(filt_im**2))

    local_maxima = extract_local_maxima(filt_im, rms * threshold)

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


def call(parent):
    res = messagebox.askquestion("Continue Program", "Do you wish to continue?")

    if res == "yes":
        messagebox.showinfo("", "Returning to program!")
        parent.destroy()
        pass

    else:
        sys.exit("Program terminating. Feel free to change the threshold!")
        parent.destroy()


def message():
    root = tk.Tk()
    root.withdraw()

    call(root)
