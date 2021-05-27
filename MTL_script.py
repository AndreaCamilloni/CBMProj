import os

from tensorflow.keras import *
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import load_model

from Dataset import train_df, test_df, valid_df, labels_cols, labels, labels_name
from Dataset.Sequence import GenericImageSequence, MultiTaskSequence
from Model import Model_XtoC
from Model.CBM import logReg
from Utils.sevenPtRule import diagnosis
from Utils.utils import export_graph, plot_confusion, report_to_csv, accuracy_on_stratiefiedKfold

import matplotlib as mpl
#mpl.rcParams.update({'font.size': 22, 'text.color': "black"})
mpl.use('Agg')
import matplotlib.pyplot as plt

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

train_gen = MultiTaskSequence(train_df, 'derm', batch_size=16,shuffle=True)
valid_gen = MultiTaskSequence(valid_df, 'derm', batch_size=16, shuffle=True,reshuffle_each_epoch=False)
test_gen = MultiTaskSequence(test_df, 'derm', batch_size=16,shuffle=False,
                                reshuffle_each_epoch=False)


early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001,  # minimium amount of change to count as an improvement
    patience=100,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

history = multi_task_model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=200,
    callbacks=[early_stopping],
    #class_weight=class_weights
)
multi_task_model.save('multitask_model.h5')
history_frame = pd.DataFrame(history.history)
history_frame.to_csv('score.csv')

for t in labels_cols:
    export_graph(_history=history, file_name=t+'accuracy.png',
                 train=t+'_categorical_accuracy',
                 val='val_'+ t +'_categorical_accuracy',
                 title=t+'accuracy', y_label='accuracy', x_label='epoch',
                 legend=['train', 'valid'])
    export_graph(_history=history, file_name=t+'loss.png', train=t+'_loss',
                 val='val_'+t+'_loss',
                 title=t+'loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])

# summarize history for loss
export_graph(_history=history, file_name='loss.png', train='loss', val='val_loss',
             title='model loss', y_label='loss', x_label='epoch', legend=['train', 'valid'])


y_pred = multi_task_model.predict(test_gen)
y_pred_train = multi_task_model.predict(train_gen)
c_hat = {}
c_hat_train = {}
for idx, t in enumerate(labels_cols):
    c_hat[idx] = np.argmax(y_pred[idx], axis=1)
    c_hat_train[idx] = np.argmax(y_pred_train[idx], axis=1)
    # plot and save CM for each concept
    plot_confusion(y_true=test_df[t], y_pred=c_hat[idx], labels=labels[idx], figsize=(6, 4))
    plt.title(labels_name[idx] + ' - dermoscopic images')
    plt.savefig('mtl/' + labels_name[idx] + '_CM.png')
    plt.close()
    # classification report to csv
    report_to_csv(c_hat[idx], test_df[t], 'mtl/' + t + '_result.csv')

# Predicted concepts for logistic regression training
predicted_concepts_df_training = pd.DataFrame(c_hat_train)
predicted_concepts_df_training.columns = labels_cols

# Predicted concepts to test logistic regression
predicted_concepts_df = pd.DataFrame(c_hat)
predicted_concepts_df.columns = labels_cols  # c_hat test df

# Final prediction(NEV_or_MEL : DIAGNOSIS) based on predicted concepts (Sequential Model: C_HAT => Y)
c_HAT_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=predicted_concepts_df_training,
                    train_label=train_df.diagnosis_numeric)
y_sequential_model = c_HAT_to_y.predict(predicted_concepts_df)

# plot and save CM for DIAG
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_sequential_model, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_SeqMod' + ' - dermoscopic images');
plt.savefig('mtl/Diagnosis/' + 'DIAG' + '_CM' + '_SeqMod')
plt.close()
# classification report to csv
report_to_csv(y_sequential_model, test_df.diagnosis_numeric, "mtl/Diagnosis/resultSeqMod.csv")
# K-FOLD
accuracy_on_stratiefiedKfold(c_HAT_to_y, predicted_concepts_df, test_df.diagnosis_numeric, "SeqMod")

# Final prediction(NEV_or_MEL : DIAGNOSIS) based on real concepts (Independent Model: true_C => Y)
c_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=train_df[labels_cols],
                train_label=train_df.diagnosis_numeric)
y_independent_model = c_to_y.predict(predicted_concepts_df)

plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_independent_model, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_IndMod' + ' - dermoscopic images');
plt.savefig('mtl/Diagnosis/' + 'DIAG' + '_CM' + '_IndMod')
plt.close()
# classification report to csv
report_to_csv(y_independent_model, test_df.diagnosis_numeric, "mtl/Diagnosis/resultIndMod.csv")
# K-FOLD
accuracy_on_stratiefiedKfold(c_to_y, predicted_concepts_df, test_df.diagnosis_numeric, "IndMod")

"""7PointChecklist"""
# 7pt based on true C
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=diagnosis(test_df, 3), labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_7ptChecklist' + ' - dermoscopic images');
plt.savefig('mtl/Diagnosis/' + 'DIAG' + '_CM' + '_7pt_trueC')
plt.close()

# classification report to csv
report_to_csv(diagnosis(test_df, 3), test_df.diagnosis_numeric, "CBM/Diagnosis/result7pt_trueC.csv")

# 7pt based on C_Hat
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=diagnosis(predicted_concepts_df, 3), labels=['NEV', 'MEL'],
               figsize=(6, 4))
plt.title('DIAG_7ptChecklist' + ' - dermoscopic images');
plt.savefig('mtl/Diagnosis/' + 'DIAG' + '_CM' + '_7pt_CHat')
plt.close()
# classification report to csv
report_to_csv(diagnosis(predicted_concepts_df, 3), test_df.diagnosis_numeric, "mtl/Diagnosis/result7pt_CHat.csv")

