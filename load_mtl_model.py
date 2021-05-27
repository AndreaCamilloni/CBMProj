import os

from tensorflow.keras import *

from tensorflow.python.keras.models import load_model
import pandas as pd
import numpy as np
from Dataset import train_df, test_df, valid_df, labels_cols, labels, labels_name
from Dataset.Sequence import GenericImageSequence, MultiTaskSequence
from Model import Model_XtoC
from Utils.utils import export_graph, plot_confusion, report_to_csv, accuracy_on_stratiefiedKfold

from Model.CBM import logReg, logReg_predict
from Utils.sevenPtRule import diagnosis
import matplotlib as mpl

# mpl.rcParams.update({'font.size': 22, 'text.color': "black"})
mpl.use('Agg')
import matplotlib.pyplot as plt


train_gen = MultiTaskSequence(train_df, 'derm', batch_size=16, shuffle=False,
                              reshuffle_each_epoch=False)
test_gen = MultiTaskSequence(test_df, 'derm', batch_size=16, shuffle=False,
                             reshuffle_each_epoch=False)
multi_task_model = load_model('multitask_model.h5')

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

