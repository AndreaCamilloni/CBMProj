import pandas as pd
import matplotlib as mpl

mpl.rcParams.update({'font.size': 22, 'text.color': "black"})
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
    plt.savefig('mtl/'+file_name)
    plt.close()
