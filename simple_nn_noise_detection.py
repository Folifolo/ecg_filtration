from keras.layers import *
from keras.models import Model

from dataset import load_good_holter
from evaluation import train_eval


def build_simple_network(input_shape):
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
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
    return autoencoder


if __name__ == "__main__":
    MODEL_PATH = "simple_detection"

    model = build_simple_network((None, 1))
    # model = load_model("models\\" + MODEL_PATH + "_ma.h5")
    model.summary()

    X = load_good_holter()
    train_eval(model, X, only_eval=False, save_path=MODEL_PATH, size=2048, epochs=50)
