# Single task model with CNN
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
from sklearn.metrics import classification_report, confusion_matrix

from Utils.utils import export_graph, plot_confusion, report_to_csv

# parameters
base_model = 'IncNet'  # base for feature extraction(IncNet, MobNet or ResNet)
lr = 0.01  # learning rate
path = '/'  # path for saving data and plots
input_shape = (256, 256, 3)
size = (256, 256)
loss_func = "FocXentropy"  # FocXentropy or CatXentropy

model = single_task_model(lr=lr, input_shape=input_shape, base=base_model, loss="FocXentropy")

""" Pre-training on HAM10000 dataset
ham10000_train_df, ham10000_test_df = train_test_split(ham10000_df, test_size=0.3)
train_gen = GenericImageSequence(ham10000_train_df, 'derm', 'diagnosis_numeric', batch_size=64,
                                  shuffle=True, image_dir = '/home/andreac/HAM10000/images/')
valid_gen = GenericImageSequence(ham10000_test_df, 'derm', 'diagnosis_numeric', batch_size=64, shuffle=True,
                                 reshuffle_each_epoch=False, image_dir = '/home/andreac/HAM10000/images/')
"""
# Keras Sequence
train_gen = GenericImageSequence(oversampling_balance_df(train_df), 'derm', 'diagnosis_numeric', batch_size=64,
                                 shuffle=True, new_size=size)
valid_gen = GenericImageSequence(test_df, 'derm', 'diagnosis_numeric', batch_size=64, shuffle=True,
                                 reshuffle_each_epoch=False, new_size=size)
test_gen = GenericImageSequence(test_df, 'derm', 'diagnosis_numeric', batch_size=64, shuffle=False,
                                reshuffle_each_epoch=False, new_size=size)


early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001,  # minimium amount of change to count as an improvement
    patience=50,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=200,
    callbacks=[early_stopping],
    # class_weight=class_weights
)

model.save(base_model + '/model.h5')

history_frame = pd.DataFrame(history.history)
history_frame.to_csv(base_model + '/score.csv')

# summarize history for accuracy
export_graph(_history=history, file_name='accuracy.png', train='categorical_accuracy', val='val_categorical_accuracy',
             title='model accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'], path=base_model)

# summarize history for loss
export_graph(_history=history, file_name='loss.png', train='loss', val='val_loss',
             title='model loss', y_label='loss', x_label='epoch', legend=['train', 'valid'], path=base_model)

Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)

plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_pred, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_SingleTask' + ' - dermoscopic images');
plt.savefig('/' + base_model + '/DIAG' + '_CM' + '_SingleTask')
plt.close()
# classification report to csv
report_to_csv(y_pred, test_df.diagnosis_numeric, path + base_model + "/resultSingleTask.csv")
