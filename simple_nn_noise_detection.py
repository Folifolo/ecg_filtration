from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dataset import *
from keras_unet.models import custom_unet
from keras.layers import *
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from dataset import *
from generators import artefact_for_detection_holter


def build_network(input_shape):
    inputs = Input(input_shape)
    x = Conv1D(64, 3, activation="relu", padding='same')(inputs)
    x = MaxPool1D(2)(x)
    x = Conv1D(32, 5, activation="relu", padding='same')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(16, 8, activation="relu", padding='same')(x)
    encoded = MaxPooling1D(2)(x)

    x = Conv1D(16, 8, activation="relu", padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 5, activation="relu", padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation="relu", padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(6, 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


if __name__ == "__main__":
    MODEL_SAVE_PATH = "models\\simple_detection.h5"
    X_train = load_holter(0)
    X_test = load_holter(1)

    size = 1024

    generator_train = artefact_for_detection_holter(X_train, size, 10, noise_type='ma')
    generator_test = artefact_for_detection_holter(X_test, size, 10, noise_type='ma')

    model = build_network((None, 1))
    plot_model(model)

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='loss', verbose=1, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    model.fit_generator(generator_train, epochs=150, steps_per_epoch=10, callbacks=callbacks_list, verbose=1)
