from keras.layers import *
from keras.models import Model
from keras_self_attention import SeqSelfAttention

from dataset import load_good_holter
from evaluation import train_eval


def build_attention_conv_network(input_shape):
    inputs = Input(input_shape)
    x = Conv1D(64, 20, activation="relu", padding='same')(inputs)
    x = MaxPool1D(2)(x)
    x = Conv1D(32, 20, activation="relu", padding='same')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(16, 20, activation="relu", padding='same')(x)
    x = MaxPooling1D(2)(x)

    x = SeqSelfAttention(attention_activation='sigmoid')(x)

    x = Conv1D(16, 20, activation="relu", padding='same')(x)
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
    MODEL_PATH = "attention_conv_detection"

    model = build_attention_conv_network((None, 1))
    # model = load_model("models\\" + MODEL_PATH + "_ma.h5")
    model.summary()

    X = load_good_holter()
    train_eval(model, X, only_eval=True, save_path=MODEL_PATH, size=2048, epochs=150)
