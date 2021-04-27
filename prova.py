import os

from tensorflow.keras import *
import pandas as pd
from tensorflow.python.keras.models import load_model

from Dataset import train_df, test_df, valid_df
from Dataset.Sequence import GenericImageSequence

reconstructed_model = load_model('model.h5')
train_gen = GenericImageSequence(train_df,'derm','diagnosis_numeric', batch_size=16)
valid_gen = GenericImageSequence(valid_df,'derm','diagnosis_numeric', batch_size=16)
test_gen = GenericImageSequence(test_df,'derm','diagnosis_numeric', batch_size=16)


early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=10, # how many epochs to wait before stopping
    restore_best_weights=True,
)


history = reconstructed_model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
    callbacks=[early_stopping]
)
reconstructed_model.save('model1.h5')
history_frame = pd.DataFrame(history.history)
history_frame.to_csv('score.csv')
