import pandas as pd
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report

# mpl.rcParams.update({'font.size': 22, 'text.color': "black"})
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.utils import resample

# from sklearn.model_selection import StratifiedKFold
# Using 5-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys


def balance_df(df):
    nev = df[df.diagnosis_numeric == 0]
    mel = df[df.diagnosis_numeric == 1]
    nev_downsampled = resample(nev,
                               replace=False,
                               n_samples=len(mel))
    return pd.concat([nev_downsampled, mel])


def oversampling_balance_df(df):
    nev = df[df.diagnosis_numeric == 0]
    mel = df[df.diagnosis_numeric == 1]
    return pd.concat([mel, nev, mel]).sample(frac=1)


def export_graph(_history, file_name, train, val, title, y_label, x_label='epoch', legend=['train', 'valid'], path=''):
    plt.plot(_history.history[train])
    plt.plot(_history.history[val])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend, loc='upper left')
    plt.savefig(path + file_name)
    plt.close()


def report_to_csv(y_pred, y_true, filename):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(filename, index=True)


def plot_confusion(y_true, y_pred, labels, fontsize=18,
                   figsize=(16, 12), cmap=plt.cm.coolwarm_r, ax=None, colorbar=True,
                   xrotation=30, yrotation=30):
    label_indexes = np.arange(0, len(labels))

    cm = confusion_matrix(y_true, y_pred, label_indexes)
    # Normalized per-class.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        # Create a new figure if no axis is specified.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    cax = ax.matshow(cm_normalized, vmin=0, vmax=1, alpha=0.8, cmap=cmap)

    # Print the number of samples that fall within each cell.
    x, y = np.meshgrid(label_indexes, label_indexes)
    for (x_val, y_val) in zip(x.flatten(), y.flatten()):
        c = cm[int(x_val), int(y_val)]
        ax.text(y_val, x_val, c, va='center', ha='center', fontsize=fontsize)

    if colorbar:
        cb = plt.colorbar(cax, fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=fontsize)

    # Make the confusion matrix pretty.
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    plt.xticks(label_indexes, rotation=xrotation)
    plt.yticks(label_indexes, rotation=yrotation, va='center', x=0.05)
    plt.ylabel('True label', fontweight='bold', fontsize=fontsize)
    plt.xlabel('Predicted label', fontweight='bold', fontsize=fontsize)
    ax.set_xticks(np.arange(-.5, len(ax.get_xticks()), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ax.get_yticks()), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.grid(b=False, which='major')

    return cax


def accuracy_on_stratiefiedKfold(model, x_test, true_label, model_name):
    stratifiedKFold = StratifiedKFold(n_splits=5)
    results = cross_val_score(model, x_test, true_label, cv=stratifiedKFold, scoring='accuracy')
    stdoutOrigin = sys.stdout
    sys.stdout = open("CBM/" + model_name + ".txt", "w")
    print(model_name + ": Results on KFold:", results,
          "Mean Accuracy: %.3f%%, Standard Deviation: (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
    sys.stdout.close()
    sys.stdout = stdoutOrigin


# Return dict of labels weight
def labels_weight(df):
    # weights for imbalance df
    NEV_count, MEL_count = np.bincount(df.diagnosis_numeric)
    total_count = len(df.diagnosis_numeric)
    weight_NEV = (1 / NEV_count) * (total_count) / 2.0
    weight_MEL = (1 / MEL_count) * (total_count) / 2.0
    class_weights = {0: weight_NEV, 1: weight_MEL}
    return class_weights
