from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import *
import pandas as pd
from Dataset import df, train_df, test_df, valid_df
from Dataset.Sequence import GenericImageSequence
from Model import single_task_model, single_task_modelprova

model = single_task_modelprova()

#train_df, test_df = train_test_split(df, test_size=0.3) #splittare secondo train e test indexes
train_gen = GenericImageSequence(train_df,'derm','diagnosis_numeric', batch_size=16)
valid_gen = GenericImageSequence(valid_df,'derm','diagnosis_numeric', batch_size=16)
test_gen = GenericImageSequence(test_df,'derm','diagnosis_numeric', batch_size=16)


early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=10, # how many epochs to wait before stopping
    restore_best_weights=True,
)


history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10,
    callbacks=[early_stopping]
)
model.save('model.h5')
#train_gen[0]
history_frame = pd.DataFrame(history.history)
history_frame.to_csv('score.csv')