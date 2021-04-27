from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import *
import pandas as pd
import numpy as np
from Dataset import df, train_df, test_df, valid_df
from Dataset.Sequence import GenericImageSequence
from Model import single_task_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

model = single_task_model()

#train_df, test_df = train_test_split(df, test_size=0.3) #splittare secondo train e test indexes
train_gen = GenericImageSequence(train_df,'derm','diagnosis_numeric', batch_size=24, shuffle=True)
valid_gen = GenericImageSequence(valid_df,'derm','diagnosis_numeric', batch_size=24, shuffle=True)
test_gen = GenericImageSequence(test_df,'derm','diagnosis_numeric', batch_size=24, shuffle=True)


early_stopping = callbacks.EarlyStopping(
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=25, # how many epochs to wait before stopping
    restore_best_weights=True,
)


history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=100,
    callbacks=[early_stopping]
)
model.save('model.h5')
#train_gen[0]
history_frame = pd.DataFrame(history.history)
history_frame.to_csv('score.csv')

# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('accuracy.png')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('loss.png')
plt.close()

Y_pred = model.predict(test_gen)
#
y_pred = np.argmax(Y_pred, axis=1)
cm=confusion_matrix(test_df.diagnosis_numeric, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.savefig('cm.png')
plt.close()

print(classification_report(test_df.diagnosis_numeric, y_pred))