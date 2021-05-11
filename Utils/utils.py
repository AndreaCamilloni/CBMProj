import pandas as pd
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report

#mpl.rcParams.update({'font.size': 22, 'text.color': "black"})
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.utils import resample


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("cm.png")
    plt.close()


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
    nev_downsampled = resample(nev,
                               replace=False,
                               n_samples=len(mel))
    return pd.concat([mel, nev_downsampled, mel]).sample(frac=1)


def export_graph(_history, file_name, train, val, title, y_label, x_label='epoch', legend=['train', 'valid']):
    plt.plot(_history.history[train])
    plt.plot(_history.history[val])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend, loc='upper left')
    plt.savefig(file_name)
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
