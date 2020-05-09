from keras.engine.saving import load_model
from keras.layers import *
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

from dataset import load_good_holter
from evaluation import get_predict_labels
from evaluation import train_eval, plot_confusion, binary_class, new_metric
from generators import artefact_for_detection_3_in_2_out


def build_support_net():
    input = Input((500, 1))
    y = Conv1D(16, 10, activation="relu")(input)
    y = MaxPool1D(2)(y)
    y = Conv1D(16, 10, activation="relu")(y)
    y = MaxPool1D(2)(y)
    y = Conv1D(32, 10, activation="relu")(y)
    y = MaxPool1D(2)(y)
    y = Conv1D(32, 10, activation="relu")(y)
    y = MaxPool1D(2)(y)
    y = Conv1D(64, 10, activation="relu")(y)
    y = Flatten()(y)
    y_out = Dense(2, activation="sigmoid")(y)
    return input, y_out


def build_triple_detection_network(input_shape):
    denum = 16
    inputs = []
    outputs = []
    inputs.append(Input(input_shape))
    inputs.append(Input((input_shape[0] //denum, input_shape[1])))
    #inputs.append(Input((500, 1)))
    kernel1 = 30
    kernel2 = 5
    dil1 = 1
    dil2 = 1

    y_input, y_out = build_support_net()
    inputs.append(y_input)

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
    x1 = UpSampling1D(denum//2)(x1)

    x = concatenate([x, x1], -1)

    y = UpSampling1D(input_shape[0] // 8)(y_out)
    y = Reshape((input_shape[0] // 8, 2))(y)
    x = concatenate([x, y], -1)

    x = Conv1D(64, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel1, activation="relu", dilation_rate=dil1, padding='same')(x)
    x = UpSampling1D(2)(x)

    out = Conv1D(6, kernel1, activation='softmax', padding='same')(x)
    outputs.append(out)
    outputs.append(y_out)
    # x = Conv1D(6, 20, activation="relu", padding='same')(x)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
    return autoencoder


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


if __name__ == "__main__":
    import numpy as np

    MODEL_PATH = "triple_detection"

    model = build_triple_detection_network((4096, 1))
    plot_model(model)
    model = load_model("models\\" + MODEL_PATH + "_ma.h5")
    model.summary()
    X = load_good_holter()

    model = train_eval(model, X, only_eval=True, save_path=MODEL_PATH, generator=artefact_for_detection_3_in_2_out,
                       size=4096, epochs=150)