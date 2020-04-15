from keras.layers import *
from keras.models import Model

from evaluation import train_eval, load_split


def build_recurrent_network(input_shape):
    inputs = Input(input_shape)
    x = Conv1D(64, 20, activation="relu", padding='same')(inputs)
    x = MaxPool1D(2)(x)
    x = Conv1D(32, 20, activation="relu", padding='same')(x)
    x = MaxPool1D(2)(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)

    x = Conv1D(32, 20, activation="relu", padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 20, activation="relu", padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(6, 20, activation='softmax', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
    return autoencoder


if __name__ == "__main__":
    MODEL_PATH = "recurrent_detection"

    model = build_recurrent_network((None, 1))
    # model = load_model(MODEL_PATH)
    model.summary()

    X = load_split()
    train_eval(model, X, only_eval=True, save_path=MODEL_PATH, size=2048, epochs=150)
