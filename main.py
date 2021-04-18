from sklearn.model_selection import train_test_split
from tensorflow import keras

from Dataset import df_og
from Dataset.Sequence import GenericImageSequence
from Model import single_task_model

model = single_task_model()
train_df, test_df = train_test_split(df_og, test_size=0.2)

train_gen = GenericImageSequence(train_df,'derm','diagnosis_numeric', batch_size=1)

train_gen[0]
