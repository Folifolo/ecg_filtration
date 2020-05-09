from keras.layers import *
from keras.models import Model

from dataset import load_good_holter
from evaluation import train_eval
from generators import artefact_for_detection_dual


def build_dual_input_network(input_shape):
    inputs = [Input(input_shape), Input(input_shape)]
    kernel1 = 30
    kernel2 = 5
    dil1 = 1
    dil2 = 1
    x = Conv1D(16, kernel1, activation="relu", dilation_rate=dil1, padding='same')(inputs[0])
    x1 = Conv1D(16, kernel2, activation="relu", dilation_rate=dil2, padding='same')(inputs[1])
    x = MaxPool1D(2)(x)
    x1 = MaxPool1D(2)(x1)
    x = Conv1D(32, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x1 = Conv1D(32, kernel2, activation="relu", dilation_rate=dil2, padding='same')(x1)
    x = MaxPool1D(2)(x)
    x1 = MaxPool1D(2)(x1)
    x = Conv1D(64, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x1 = Conv1D(64, kernel2, activation="relu", dilation_rate=dil2, padding='same')(x1)
    x = MaxPool1D(2)(x)
    x1 = UpSampling1D(4)(x1)

    x = concatenate([x, x1], -1)

    x = Conv1D(64, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x = UpSampling1D(2)(x)

    x = Conv1D(6, kernel1, activation='softmax', padding='same')(x)

    autoencoder = Model(inputs, x)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
    return autoencoder


if __name__ == "__main__":
    MODEL_PATH = "dual_input_detection"

    model = build_dual_input_network((None, 1))
    # model = load_model("models\\" + MODEL_PATH + "_ma.h5")
    model.summary()

    X = load_good_holter()
    train_eval(model, X, only_eval=False, save_path=MODEL_PATH, generator=artefact_for_detection_dual,
               size=4096, epochs=100)
