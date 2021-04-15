from sklearn.model_selection import train_test_split
from tensorflow import keras

from Dataset import mtl_df_t
from Dataset.Sequence import GenericImageSequence

base_model = keras.applications.InceptionV3(weights=None, classes=1)
base_model.summary()
base_model.compile(
    loss={
        'dermatofibroma': 'binary_crossentropy'
    },
    optimizer='adam',
    metrics=['accuracy']

)
train_df, test_df = train_test_split(mtl_df_t, test_size=0.2)

train_gen = GenericImageSequence(train_df,'derm','0', batch_size=1)

train_gen[0]