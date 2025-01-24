from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_cm(y_test_all: list, y_pred_all: list, labels: list[str]) -> None:
    # get the matrix as counts and percents
    count = confusion_matrix(y_test_all, y_pred_all)
    per = confusion_matrix(y_test_all, y_pred_all, normalize="true")

    # create dataframe
    cm = pd.DataFrame(per, index=labels, columns=labels)
    cm.index.name = 'True Labels'
    cm.columns.name = 'Predicted Labels'

    # plot
    annot = np.asarray([f"{v1:,d}\n({v2:.0%})" for v1, v2 in zip(count.ravel(), per.ravel())])
    annot = annot.reshape((len(labels), len(labels)))

    plt.figure(figsize=(6, 10), dpi=200)
    sns.set(font_scale=1.)
    sns.heatmap(cm, annot=annot, cbar=False, fmt='', cmap="Blues", linewidths=0.1, square=True, vmin=0, vmax=1,
                linecolor="Black")
    plt.ylabel('True Labels', weight="bold")
    plt.xlabel('Predicted Labels', weight="bold")
    plt.show()
