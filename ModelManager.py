from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import numpy as np

BATCH_SIZE = 32
EPOCHS = 20
TEST_SIZE = 0.3

def train_network(model, inputs, outputs):
    # Dados já vem normalizados
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=TEST_SIZE)
    # x_train = x_train[..., np.newaxis]
    # x_test = x_test[..., np.newaxis]
    # y_train = y_train[..., np.newaxis]
    # y_test = y_test[..., np.newaxis]

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.summary()
    return model, x_test


def train_vae(vae, inputs, outputs):
    # Dados já vem normalizados
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=TEST_SIZE)
    # x_train = x_train[..., np.newaxis]
    # x_test = x_test[..., np.newaxis]
    # y_train = y_train[..., np.newaxis]
    # y_test = y_test[..., np.newaxis]

    vae.model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)
    vae.model.summary()
    return vae, x_test

def save_model(model, name):
    save_path = 'models/' + name
    #tf.keras.models.save_model(model, save_path)
    model.save(save_path)


def load_model(model_name):
    return tf.keras.models.load_model('models/' + model_name)


def save_scaler(scaler, name='scaler'):
    save_path = 'minmax/' + name + '.gz'
    joblib.dump(scaler, save_path)
    # my_scaler = joblib.load('scaler.gz')
    pass


def load_scaler(name='scaler'):
    load_path = 'minmax/' + name + '.gz'
    return joblib.load(load_path)
