# Analysis for the Application of Machine Learning to the EIC Synchrotron Radiation
# Main imports
import tensorboard
from keras.backend import dropout, sparse_categorical_crossentropy, categorical_crossentropy
import tensorflow.keras as keras
import tensorflow as tf
import sklearn as skl
import numpy as np

from time import localtime

from load_data import load_data
from keras import activations, layers
from keras import models
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, StandardScaler, MinMaxScaler
import keras_tuner 
from keras.metrics import AUC
from keras import activations

fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-NoAU.events.root"; oname = "NoAU"
# fname = "data/SR-PHOTONS-LUND-NOFMT-20bun-for-Andrey-AU.events.root"; oname = "AU"
X_train, X_test, y_train, y_test, rdf = load_data(fname, 1/2, add_cols=True, discard_data=True, upsample=True)#, 

for i in np.unique(y_train):
    print(i, y_train[y_train == i].shape)

# lb = MultiLabelBinarizer()
lb = LabelBinarizer() 
lb.fit(sorted(y_train))

scaler = MinMaxScaler(feature_range=(-1,1)) #StandardScaler()#
scaler.fit(X_train)

def my_model(input_dim, n_outputs):
    n_layers = 3
    units = 128
    dropout = 0.1

    model = models.Sequential()
    
    model.add(layers.Dense(units, activation='relu', input_dim=input_dim))
    # model.add(layers.Dropout(dropout))
    
    for layer in range(n_layers-1):
        model.add(layers.Dense(units, activation='relu'))
        # model.add(layers.Dropout(dropout))

    model.add(layers.Dense(n_outputs, activation='sigmoid'))
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    return model


def my_hp_model(hp, input_dim=X_train.shape[1], n_outputs=len(lb.classes_)):
    n_layers = hp.Int('layers', min_value=1, max_value=8, step=1)
    units    = hp.Choice('units', [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024])
    # (norm, dropout)  = hp.Choice('norm_dropout', [(True, 0.), (False, 0.), (False, 0.1), (False, 0.2), (False, 0.3)])
    dropout  = hp.Choice('dropout', [0., 0.1, 0.2, 0.3])
    # norm     = hp.Choice('batch_norm', [True, False]) 
    activate = hp.Choice('activation', ["relu","leaky_relu"])
    fin_act = hp.Choice('final_act', ['sigmoid', 'softmax'])
    
    model = models.Sequential()
    model.add(layers.InputLayer(input_dim))
    for l in range(n_layers+1):
        model.add(layers.Dense(units))
        if not dropout:
            model.add(layers.BatchNormalization())

        if activate == "relu":
            model.add(layers.ReLU())
        else:
            model.add(layers.LeakyReLU())

        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(n_outputs, activation=fin_act))
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    return model
    
t = localtime()
opath = f"DNN_{t.tm_mon:02d}{t.tm_mday:02d}{t.tm_year-2000:02d}_{t.tm_hour:02d}{t.tm_min:02d}"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='models/DNN/'+opath,
        save_freq='epoch', period=8,
        save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, restore_best_weights=True),
    keras.callbacks.TensorBoard()
]

X_train_transformed = scaler.transform(X_train)
y_train_transformed = lb.transform(y_train)
X_test_transformed = scaler.transform(X_test)
y_test_transformed = lb.transform(y_test)


##########################################################################################################
# n_layers = 5, units = 256, dropout = 0.1
# w/o scaled input 100 epochs validation 3.8105416297912598
# w/  scaled input 100 epochs train 2.9320 ---- 
#  100 epochs: Test loss: 4.018465995788574 / Test accuracy: 0.2242424190044403 / Test AUC: 0.7341789603233337, 
# 1000 epochs training: - loss: 2.4333 - accuracy: 0.2955 - auc: 0.9482
# w/ batch normalization
# 10000 epochs training - loss: 2.4527 - accuracy: 0.2948 - auc: 0.9477
#
#
# n_layers = 5, units = 256, dropout = 0.1
# w/o extra features, all in lay 0 are removed, scaled input, batch nom
# 1000 epochs: loss: 3.0508 - accuracy: 0.1442 - auc: 0.9169
#   Test loss: 5.897046089172363 / Test accuracy: 0.04721567779779434 / Test AUC: 0.605785608291626
# w/  extra features, all in lay 1 are removed, scaled input
# 1000 epochs: loss: 2.8997 - accuracy: 0.1634 - auc: 0.9296
#   Test loss: 6.15309476852417 / Test accuracy: 0.04736604541540146 / Test AUC: 0.6035714745521545
#
#
# n_layers = 5, units = 256, dropout = 0.2
# w/  extra features, all in lay 0 are removed, scaled input, batch nom
# 1000 epochs: loss: 2.8997 - accuracy: 0.1634 - auc: 0.9296
#   Test loss: 5.3397135734558105 / Test accuracy: 0.046513959765434265 / Test AUC: 0.6171226501464844
#
##########################################################################################################
# 
# hyperparam tuning with only layers no BE or stave --- 250epochs each
# 7layers, 320units, 0.2dropout
# Best val_loss So Far: 1.3921701908111572
# Test loss: 1.3907777070999146 / Test accuracy: 0.37727102637290955 / Test AUC: 0.7231901288032532
# 
##########################################################################################################
# 
# hyperparam tuning with only layers --- 250epochs each
# 3layers, 64units, 0.1dropout
# Best val_loss So Far: 1.3921701908111572
# Test loss: 1.2862322330474854 / Test accuracy: 0.4451243281364441 / Test AUC: 0.8164054155349731
# 
##########################################################################################################
# 
# hyperparam tuning with only layers --- 250epochs each
# 2layers, 72units, 0.3dropout
# Best val_loss So Far: 1.2837144136428833
# Test loss: 1.281967043876648 / Test accuracy: 0.4468725621700287 / Test AUC: 0.8182138800621033
# 
##########################################################################################################

train_size = int(3/4*len(y_train))
batch_size = 32
epochs = 64
tuner = keras_tuner.RandomSearch(
    my_hp_model,
    objective='val_loss',
    max_trials=8192)

tuner.search(X_train_transformed[:train_size, :], y_train_transformed[:train_size], 
            validation_data=(X_train_transformed[train_size:, :], y_train_transformed[train_size:]),
            batch_size=batch_size, epochs=epochs, callbacks=callbacks)


model = tuner.get_best_models()[0]

###########################################################################################################

# with downsampled lay0 to max of other layers and 3 layers of 64 each and 0.1 dropout
# Test loss: 1.241578221321106 / Test accuracy: 0.46937745809555054 / Test AUC: 0.8331909775733948

# with all samples at equal priors
# Test loss: 1.330015778541565 / Test accuracy: 0.4305014908313751 / Test AUC: 0.8071685433387756


# model = my_model(X_train.shape[1], len(lb.classes_))
epochs = 512
batch_size = 16

model.summary()
plot_model(model, to_file='imgs/generator_plot.png', show_shapes=True, show_layer_names=True)

try:
    model.fit(X_train_transformed, y_train_transformed, validation_data=(X_test_transformed, y_test_transformed), 
            batch_size=batch_size, epochs=epochs, callbacks=callbacks)
except KeyboardInterrupt:
    pass

###########################################################################################################

score = model.evaluate(X_test_transformed, y_test_transformed, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]} / Test AUC: {score[2]}')

t = localtime()
model.save(f"models/DNN/"+opath)
