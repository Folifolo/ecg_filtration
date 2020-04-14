import matplotlib.pyplot as plt
from keras.engine.saving import load_model
from keras.layers import *
from keras.models import Model
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from evaluation import train_eval_ecg
from generators import artefact_for_detection_ecg
from see_rnn import get_layer_outputs, show_features_1D


def build_network(input_shape):
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
    model = build_network((2048, 1))
    model.summary()
    model = load_model("models\\recurrent_detection_ma.h5")

    train_eval_ecg(model, should_eval=True, should_load=False, MODEL_SAVE_PATH="models\\attention_detection_ma.h5")
    X = load_dataset()['x']
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
    noise_prob = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5]
    gener = artefact_for_detection_ecg(X_train, 2048, 10, noise_type='ma', noise_prob=noise_prob)
    X = next(gener)[0]
    outs = get_layer_outputs(model, X, layer_idx=5)

    outs1 = get_layer_outputs(model, X, layer_idx=6)
    for i in range(10):
        plt.plot(X[i])
        plt.show()
        show_features_1D(outs[i:i + 1], n_rows=8, show_borders=False)
        show_features_1D(outs1[i:i + 1], n_rows=8, show_borders=False)
