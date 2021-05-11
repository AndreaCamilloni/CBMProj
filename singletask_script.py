import os

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import *
import pandas as pd
import numpy as np
from Dataset import *
from Dataset.Sequence import GenericImageSequence
from Dataset.ham10000 import ham10000_df
from Model import single_task_model
import matplotlib as mpl

mpl.use('Agg')
#mpl.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from Utils.utils import plot_confusion_matrix, export_graph, plot_confusion, report_to_csv


model = single_task_model()

# train_df, test_df = train_test_split(df, test_size=0.3) #splittare secondo train e test indexes
train_gen = GenericImageSequence(ham10000_df, 'derm', 'diagnosis_numeric', batch_size=24,
                                  shuffle=True, image_dir = '/home/andreac/HAM10000/images/')
valid_gen = GenericImageSequence(valid_df, 'derm', 'diagnosis_numeric', batch_size=24, shuffle=True,
                                 reshuffle_each_epoch=False)
test_gen = GenericImageSequence(test_df, 'derm', 'diagnosis_numeric', batch_size=24, shuffle=True,
                                reshuffle_each_epoch=False)

# weights for imbalance df
""" 
NEV_count, MEL_count = np.bincount(train_df.diagnosis_numeric)
total_count = len(train_df.diagnosis_numeric)
weight_NEV = (1 / NEV_count) * (total_count) / 2.0
weight_MEL = (1 / MEL_count) * (total_count) / 2.0
class_weights = {0: weight_NEV, 1: weight_MEL}
"""

early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001,  # minimium amount of change to count as an improvement
    patience=25,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=100,
    callbacks=[early_stopping],
    # class_weight=class_weights
)
model.save('model.h5')
# train_gen[0]
history_frame = pd.DataFrame(history.history)
history_frame.to_csv('score.csv')

# summarize history for accuracy
export_graph(_history=history, file_name='accuracy.png', train='categorical_accuracy', val='val_categorical_accuracy',
             title='model accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for loss
export_graph(_history=history, file_name='loss.png', train='loss', val='val_loss',
             title='model loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])


Y_pred = model.predict(test_gen)
#
y_pred = np.argmax(Y_pred, axis=1)

plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_pred, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_SingleTask' + ' - dermoscopic images');
plt.savefig('DIAG' + '_CM' + '_SingleTask')
plt.close()
# classification report to csv
report_to_csv(y_pred, test_df.diagnosis_numeric, "resultSingleTask.csv")
