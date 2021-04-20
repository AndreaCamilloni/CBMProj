from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import *

from Dataset import df
from Dataset.Sequence import GenericImageSequence
from Model import single_task_model

model = single_task_model()
df['diagnosis_numeric']=df['diagnosis_numeric'].apply(lambda x: 1 if x == 2 else 0) #MEL or NOT-MEL 'mapping'

train_df, test_df = train_test_split(df, test_size=0.3) #splittare secondo train e test indexes
train_gen = GenericImageSequence(train_df,'derm','diagnosis_numeric', batch_size=16)
test_gen = GenericImageSequence(test_df,'derm','diagnosis_numeric', batch_size=16)


early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=10, # how many epochs to wait before stopping
    restore_best_weights=True,
)


history = model.fit(
    train_gen,
    validation_data=test_gen,
    steps_per_epoch=30,
    validation_steps=10,
    epochs=5)

#train_gen[0]
