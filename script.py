""" CBM with logistic regression as classifier + 7pt algorithm """
import os
import pandas as pd
from Dataset import image_dir, train_df, test_df, labels_cols, labels_name, labels
from Model.CBM import deep_features, logReg, logReg_predict
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras

import matplotlib as mpl

from Utils.sevenPtRule import diagnosis
from Utils.utils import plot_confusion, report_to_csv, accuracy_on_stratiefiedKfold

mpl.use('Agg')
import matplotlib.pyplot as plt

# base Model with weights pre-trained on imagenet: ResNet101V2, MobileNet, InceptionV3
input_shape = (256, 256, 3)
base_model = keras.applications.ResNet101V2(include_top=False, input_shape=input_shape)



# Features extraction from image inputs
train_derm_f = deep_features(img_paths=[os.path.join(image_dir, img_path) for img_path in train_df.derm],
                             model=base_model,
                             func_preprocess_input=preprocess_input,
                             target_size=(256, 256, 3), )
test_derm_f = deep_features(img_paths=[os.path.join(image_dir, img_path) for img_path in test_df.derm],
                            model=base_model,
                            func_preprocess_input=preprocess_input,
                            target_size=(256, 256, 3), )

# SINGLE TASK DIAGNOSIS X => Y
x_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=train_derm_f,
                train_label=train_df.diagnosis_numeric)
y_singletask = x_to_y.predict(test_derm_f)
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_singletask, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_x_to_y' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_x_to_y')
plt.close()
# classification report to csv
report_to_csv(y_singletask, test_df.diagnosis_numeric, "CBM/Diagnosis/result_x_to_y.csv")
# accuracy on stratifiedKfold(n_spilts=5)
accuracy_on_stratiefiedKfold(x_to_y,test_derm_f,test_df.diagnosis_numeric,"x_to_y")


# Concepts extraction from features (X=>C_HAT)
# (X=>C_HAT) valid for both model, independent and sequential
x_to_C_HAT = {}
C_HAT_test = {}
C_HAT_train = {}
for idx, t in enumerate(labels_cols):
    x_to_C_HAT[idx] = logReg(class_weight='balanced', train_inputs=train_derm_f, train_label=train_df[t], C=0.01)
    C_HAT_test[idx] = logReg_predict(reg=x_to_C_HAT[idx], test_inputs=test_derm_f)
    C_HAT_train[idx] = logReg_predict(reg=x_to_C_HAT[idx], test_inputs=train_derm_f)
    # plot and save CM for each concept
    plot_confusion(y_true=test_df[t], y_pred=C_HAT_test[idx], labels=labels[idx], figsize=(6, 4))
    plt.title(labels_name[idx] + ' - dermoscopic images')
    plt.savefig('CBM/PredictedConcepts/' + labels_name[idx] + '_CM.png')
    plt.close()
    # classification report to csv
    report_to_csv(C_HAT_test[idx], test_df[t], 'CBM/PredictedConcepts/' + t + '_result.csv')
    # accuracy on stratifiedKfold(n_spilts=5) for each concept
    accuracy_on_stratiefiedKfold(x_to_C_HAT[idx], test_derm_f, test_df[t], "x_to_" + t + "_HAT")

# Predicted concepts for logistic regression training
predicted_concepts_df_training = pd.DataFrame(C_HAT_train)
predicted_concepts_df_training.columns = labels_cols

# Predicted concepts to test logistic regression
predicted_concepts_df = pd.DataFrame(C_HAT_test)
predicted_concepts_df.columns = labels_cols  # c_hat test df

# Final prediction(NEV_or_MEL : DIAGNOSIS) based on predicted concepts (Sequential Model: C_HAT => Y)
c_HAT_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=predicted_concepts_df_training,
                    train_label=train_df.diagnosis_numeric)
y_sequential_model = c_HAT_to_y.predict(predicted_concepts_df)
# plot and save CM for DIAG
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_sequential_model, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_SeqMod' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_SeqMod')
plt.close()
# classification report to csv
report_to_csv(y_sequential_model, test_df.diagnosis_numeric, "CBM/Diagnosis/resultSeqMod.csv")
# accuracy on stratifiedKfold(n_spilts=5)
accuracy_on_stratiefiedKfold(c_HAT_to_y,predicted_concepts_df,test_df.diagnosis_numeric,"SeqMod")


# Final prediction(NEV_or_MEL : DIAGNOSIS) based on real concepts (Independent Model: true_C => Y)
c_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=train_df[labels_cols],
                train_label=train_df.diagnosis_numeric)
y_independent_model = c_to_y.predict(predicted_concepts_df)
accuracy_on_stratiefiedKfold(c_to_y,predicted_concepts_df,test_df.diagnosis_numeric,"IndMod")
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_independent_model, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_IndMod' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_IndMod')
plt.close()
# classification report to csv
report_to_csv(y_independent_model, test_df.diagnosis_numeric, "CBM/Diagnosis/resultIndMod.csv")
# accuracy on stratifiedKfold(n_spilts=5)
accuracy_on_stratiefiedKfold(c_to_y,predicted_concepts_df,test_df.diagnosis_numeric,"IndMod")

"""7PointChecklist"""
# 7pt based on true C
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=diagnosis(test_df, 3), labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_7ptChecklist' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_7pt_trueC')
plt.close()

# classification report to csv
report_to_csv(diagnosis(test_df, 3), test_df.diagnosis_numeric, "CBM/Diagnosis/result7pt_trueC.csv")

# 7pt based on C_Hat
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=diagnosis(predicted_concepts_df, 3), labels=['NEV', 'MEL'],
               figsize=(6, 4))
plt.title('DIAG_7ptChecklist' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_7pt_CHat')
plt.close()
# classification report to csv
report_to_csv(diagnosis(predicted_concepts_df, 3), test_df.diagnosis_numeric, "CBM/Diagnosis/result7pt_CHat.csv")
