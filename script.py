import os
import pandas as pd
from sklearn.metrics import classification_report

from Dataset import image_dir, train_df, test_df, labels_cols, labels_name, labels
from Model.CBM import deep_features, logReg, logReg_predict
from keras.applications.inception_v3 import InceptionV3, preprocess_input

import matplotlib as mpl

from Utils.utils import plot_confusion, report_to_csv

mpl.use('Agg')
import matplotlib.pyplot as plt

input_shape = (256, 256, 3)
base_model = InceptionV3(include_top=False, input_shape=input_shape)

# Features extraction from image inputs
train_derm_f = deep_features(img_paths=[os.path.join(image_dir, img_path) for img_path in train_df.derm],
                             model=base_model,
                             func_preprocess_input=preprocess_input,
                             target_size=(256, 256, 3), )
test_derm_f = deep_features(img_paths=[os.path.join(image_dir, img_path) for img_path in test_df.derm],
                            model=base_model,
                            func_preprocess_input=preprocess_input,
                            target_size=(256, 256, 3), )

# Concepts extraction from features (X=>C_HAT)
# (X=>C_HAT) valid for both model, independent and sequential
x_to_C_HAT = {}
C_HAT = {}
for idx, t in enumerate(labels_cols):
    x_to_C_HAT[idx] = logReg(class_weight='balanced', train_inputs=train_derm_f, train_label=train_df[t], C=0.01).fit()
    C_HAT[idx] = logReg_predict(reg=x_to_C_HAT[idx], test_inputs=test_derm_f)
    # plot and save CM for each concept
    plot_confusion(y_true=test_df[t], y_pred=C_HAT[idx], labels=labels[idx], figsize=(6, 4))
    plt.title(labels_name[idx] + ' - dermoscopic images')
    plt.savefig('CBM/PredictedConcepts/' + labels_name[idx] + '_CM')
    plt.close()
    # classification report to csv
    report_to_csv(C_HAT[idx], test_df[t], 'CBM/PredictedConcepts/'+t+'_result.csv')

predicted_concepts_df = pd.DataFrame(C_HAT)
predicted_concepts_df.columns = labels_cols

# Final prediction(NEV_or_MEL : DIAGNOSIS) based on predicted concepts (Sequential Model: C_HAT => Y)
c_HAT_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=predicted_concepts_df,
                    train_label=test_df.diagnosis_numeric).fit()
y_sequential_model = c_HAT_to_y.predict(
    test_df[labels_cols])  # la predizione finale devo farla sui concetti reali o su quelli trovati
# plot and save CM for DIAG
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_sequential_model, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_SeqMod' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_SeqMod')
plt.close()
# classification report to csv
report_to_csv(y_sequential_model, test_df.diagnosis_numeric, "CBM/Diagnosis/resultSeqMod.csv")


# Final prediction(NEV_or_MEL : DIAGNOSIS) based on real concepts (Independent Model: true_C => Y)
c_to_y = logReg(C=0.01, class_weight='balanced', train_inputs=train_df[labels_cols],
                train_label=train_df.diagnosis_numeric).fit()
y_independent_model = c_to_y.predict(predicted_concepts_df)  ### test_df[labels_cols] ### da verificare meglio
plot_confusion(y_true=test_df.diagnosis_numeric, y_pred=y_independent_model, labels=['NEV', 'MEL'], figsize=(6, 4))
plt.title('DIAG_IndMod' + ' - dermoscopic images');
plt.savefig('CBM/Diagnosis/' + 'DIAG' + '_CM' + '_IndMod')
plt.close()
# classification report to csv
report_to_csv(y_independent_model, test_df.diagnosis_numeric, "CBM/Diagnosis/resultIndMod.csv")
