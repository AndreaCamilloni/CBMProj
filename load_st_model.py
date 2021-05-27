import sys

import keras
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import *

import numpy as np
from Dataset import *
from Dataset.Sequence import GenericImageSequence
from Dataset.ham10000 import ham10000_df
from Model import single_task_model
import matplotlib as mpl

mpl.use('Agg')
# mpl.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from Utils.utils import export_graph, plot_confusion, report_to_csv
from Utils.focal_xentropy import FocalCategoricalCrossEntropy

# parameters
base_model = 'IncNet'  # base for feature extraction(IncNet, MobNet or ResNet)
lr = 0.01  # learning rate
path = '/'  # path for saving data and plots
input_shape = (256, 256, 3)
size = (256, 256)
loss_func = "FocXentropy"  # FocXentropy or CatXentropy

if loss_func == "FocXentropy":
    model = keras.models.load_model(base_model + '/model.h5',
                                    custom_objects={'FocalCategoricalCrossEntropy': FocalCategoricalCrossEntropy})
else:
    model = keras.models.load_model(base_model + '/model.h5')

test_gen = GenericImageSequence(test_df, 'derm', 'diagnosis_numeric', batch_size=64, shuffle=False,
                                reshuffle_each_epoch=False,  new_size=size)

Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
# confusion matrix
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_pred, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_SingleTask' + ' - dermoscopic images');
plt.savefig(base_model + '/DIAG' + '_CM' + '_SingleTask')
plt.close()
# classification report to csv
report_to_csv(y_pred, test_df.diagnosis_numeric, base_model + "/resultSingleTask.csv")

stratifiedKFold = StratifiedKFold(n_splits=5)
cvscores = []
for train, test in stratifiedKFold.split(test_df, test_df.diagnosis_numeric):
    test_gen = GenericImageSequence(test_df.iloc[test], 'derm', 'diagnosis_numeric', batch_size=64, shuffle=False,
                                    reshuffle_each_epoch=False, new_size=size)
    scores = model.evaluate(test_gen)
    cvscores.append(scores[1] * 100)

stdoutOrigin = sys.stdout
sys.stdout = open(base_model + "/log.txt", "w")
print(": Results on KFold:", cvscores,
      "Mean Accuracy: %.3f%%, Standard Deviation: (%.3f%%)" % (np.mean(cvscores), np.std(cvscores)))
sys.stdout.close()
sys.stdout = stdoutOrigin
