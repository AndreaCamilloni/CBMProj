import os

from tensorflow.keras import *
import pandas as pd
from tensorflow.python.keras.models import load_model

from Dataset import train_df, test_df, valid_df
from Dataset.Sequence import GenericImageSequence, MultiTaskSequence
from Model import Model_XtoC
from Utils.utils import export_graph

loss_list = ["categorical_crossentropy", 'categorical_crossentropy',
             'categorical_crossentropy', 'categorical_crossentropy',
             "categorical_crossentropy", 'categorical_crossentropy',
             'categorical_crossentropy']

test_metrics = {'pigment_network_numeric': 'categorical_accuracy', 'blue_whitish_veil_numeric': 'categorical_accuracy',
                'vascular_structures_numeric': 'categorical_accuracy', 'pigmentation_numeric': 'categorical_accuracy',
                'streaks_numeric': 'categorical_accuracy', 'dots_and_globules_numeric': 'categorical_accuracy',
                'regression_structures_numeric': 'categorical_accuracy', }

dd = 0.0

multi_task_model = Model_XtoC(loss_list,test_metrics,dd)
train_gen = MultiTaskSequence(train_df, 'derm', batch_size=16)
valid_gen = MultiTaskSequence(valid_df, 'derm', batch_size=16)
test_gen = MultiTaskSequence(test_df, 'derm', batch_size=16)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001,  # minimium amount of change to count as an improvement
    patience=10,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

history = multi_task_model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=100,
    callbacks=[early_stopping]
)
multi_task_model.save('model1.h5')
history_frame = pd.DataFrame(history.history)
history_frame.to_csv('score.csv')

# summarize history for accuracy
export_graph(_history=history, file_name='pigment_network_numeric_categorical_accuracy.png', train='pigment_network_numeric_categorical_accuracy', val='val_pigment_network_numeric_categorical_accuracy',
             title='pigment_network_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for accuracy
export_graph(_history=history, file_name='blue_whitish_veil_numeric_categorical_accuracy.png', train='blue_whitish_veil_numeric_categorical_accuracy', val='val_blue_whitish_veil_numeric_categorical_accuracy',
             title='blue_whitish_veil_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for accuracy
export_graph(_history=history, file_name='vascular_structures_numeric_categorical_accuracy.png', train='vascular_structures_numeric_categorical_accuracy', val='val_vascular_structures_numeric_categorical_accuracy',
             title='vascular_structures_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for accuracy
export_graph(_history=history, file_name='pigmentation_numeric_categorical_accuracy.png', train='pigmentation_numeric_categorical_accuracy', val='val_pigmentation_numeric_categorical_accuracy',
             title='pigmentation_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for accuracy
export_graph(_history=history, file_name='streaks_numeric_categorical_accuracy.png', train='streaks_numeric_categorical_accuracy', val='val_streaks_numeric_categorical_accuracy',
             title='streaks_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for accuracy
export_graph(_history=history, file_name='dots_and_globules_numeric_categorical_accuracy.png', train='dots_and_globules_numeric_categorical_accuracy', val='val_dots_and_globules_numeric_categorical_accuracy',
             title='dots_and_globules_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for accuracy
export_graph(_history=history, file_name='regression_structures_numeric_categorical_accuracy.png', train='regression_structures_numeric_categorical_accuracy', val='val_regression_structures_numeric_categorical_accuracy',
             title='regression_structures_numeric_categorical_accuracy', y_label='accuracy', x_label='epoch', legend=['train', 'valid'])

# summarize history for loss
export_graph(_history=history, file_name='loss.png', train='loss', val='val_loss',
             title='model loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='pigment_network_loss.png', train='pigment_network_numeric_loss', val='val_pigment_network_numeric_loss',
             title='pigment network loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='vascular_structures_loss.png', train='vascular_structures_numeric_loss', val='val_vascular_structures_numeric_loss',
             title='vascular structures numeric loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='loss.png', train='loss', val='val_loss',
             title='model loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='pigmentation_numeric_loss.png', train='pigmentation_numeric_loss', val='val_pigmentation_numeric_loss',
             title='pigmentation loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='streaks_numeric_loss.png', train='streaks_numeric_loss', val='val_streaks_numeric_loss',
             title='streaks loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='dots_and_globules_numeric_loss.png', train='dots_and_globules_numeric_loss', val='val_dots_and_globules_numeric_loss',
             title='dots and globules loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

export_graph(_history=history, file_name='regression_structures_loss.png', train='regression_structures_numeric_loss', val='val_regression_structures_numeric_loss',
             title='regression structures loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])
