from keras.layers import *
from keras.models import Model

from dataset import *
from evaluation import train_eval, load_split


def build_residual_network(input_shape):
    input_ecg = Input(shape=input_shape)

    x = Conv1D(64, 20, padding='same', activation='relu')(input_ecg)
    x = BatchNormalization()(x)

    x = residual_block(4, x)
    x = MaxPool1D(2)(x)

    x = Conv1D(32, 20, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)

    x = Conv1D(16, 20, padding='same', activation='relu')(x)
    encoder = BatchNormalization()(x)

    x = UpSampling1D(2)(encoder)
    x = Conv1D(32, 20, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling1D(2)(x)
    x = residual_block(4, x)

    decoder = Conv1D(6, 20, padding='same', activation='softmax')(x)

    residual_model = Model(input_ecg, decoder)
    residual_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return residual_model


def residual_block(size, input_layer, conv_size=20, filters_num=20):
    for i in np.arange(1, size):
        dil = [i ** 2]
        x = Conv1D(filters_num, conv_size, dilation_rate=dil, padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Conv1D(filters_num, conv_size, dilation_rate=dil, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        input_layer = Concatenate(axis=-1)([x, input_layer])
    return input_layer


if __name__ == "__main__":
    MODEL_PATH = "dense_detection"

    model = build_residual_network((None, 1))
    # model = load_model(MODEL_PATH)
    model.summary()

    X = load_split()
    train_eval(model, X, only_eval=True, save_path=MODEL_PATH, size=2048, epochs=150)
