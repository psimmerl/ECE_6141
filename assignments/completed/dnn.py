from keras.backend import print_tensor
import tensorflow.keras as keras
import tensorflow as tf
import sklearn as skl
import pandas as pd
import numpy as np

from keras import layers
from keras import models
import keras_tuner

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelBinarizer

magic_raw = pd.read_csv("datasets/magic04.data", header=None)
magic_raw.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
print("Contains nan? ", magic_raw.isnull().values.any())

split = StratifiedShuffleSplit(n_splits=1, test_size=0.75, random_state=42)
for train_index, test_index in split.split(magic_raw, magic_raw["class"]):
    magic = magic_raw.loc[train_index]
    magic_test = magic_raw.loc[test_index]


X_train, y_train = magic.drop(columns='class').to_numpy(), magic['class'].to_numpy()
X_test, y_test = magic_test.drop(columns='class').to_numpy(), magic_test['class'].to_numpy()

# magicDNN = models.Sequential([
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ], 'magicDNN')

# magicDNN.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

batch_size = 16
epochs = 100

scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)

# lb = LabelBinarizer()
y_train_transformed = np.array([1 if v =='h' else 0 for v in y_train])



callbacks = [
    # keras.callbacks.ModelCheckpoint(
    #     filepath='models/keras/model_{epoch}',
    #     save_freq='epoch',# period=200,
    #     ),#save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# magicDNN.fit(X_train_transformed, y_train_transformed, validation_split=1/3, 
#             batch_size=batch_size, epochs=epochs, callbacks=callbacks)

X_test_transformed = scaler.transform(X_test)
# y_train_transformed = lb.transform(y_test)
y_test_transformed = np.array([1 if v =='h' else 0 for v in y_test])

# score = magicDNN.evaluate(X_test_transformed, y_train_transformed, verbose=0)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]} / Test AUC: {score[2]}')




train_size = int(4/5*len(y_train))


def my_hp_model(hp, input_dim=X_train.shape[1], n_outputs=1):
    units    = hp.Int('units', min_value=1, max_value=1000, step=1)
    
    model = models.Sequential()
    model.add(layers.Dense(units, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(n_outputs, activation='sigmoid'))

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

tuner = keras_tuner.RandomSearch(
    my_hp_model,
    objective='val_accuracy',
    max_trials=1000)

print(X_train.shape, y_train.shape, train_size)
print(X_train_transformed.shape, y_train_transformed.shape, train_size)
tuner.search(X_train_transformed[:train_size, :], y_train_transformed[:train_size], 
            validation_data=(X_train_transformed[train_size:, :], y_train_transformed[train_size:]),
            batch_size=batch_size, epochs=epochs, callbacks=callbacks)

model = tuner.get_best_models()[0]



score = model.evaluate(X_test_transformed, y_train_transformed, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]} / Test AUC: {score[2]}')
