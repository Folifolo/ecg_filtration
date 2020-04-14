from keras.layers import *
from keras.models import Model

from evaluation import train_eval_holter


def build_network(input_shape):
    inputs = Input(input_shape)
    x = Conv1D(64, 20, activation="relu", padding='same')(inputs)
    x = MaxPool1D(2)(x)
    x = Conv1D(32, 20, activation="relu", padding='same')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(16, 20, activation="relu", padding='same')(x)
    encoded = MaxPooling1D(2)(x)

    x = Conv1D(16, 20, activation="relu", padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 20, activation="relu", padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 20, activation="relu", padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(6, 20, activation='softmax', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


if __name__ == "__main__":
    model = build_network((None, 1))
    train_eval_holter(model, should_eval=True, should_load=True, MODEL_SAVE_PATH = "models\\simple_detection_em.h5")