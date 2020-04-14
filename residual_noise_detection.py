from keras.layers import *
from keras.models import Model

from dataset import *
from evaluation import train_eval_holter


def build_network(input_shape):
    input_ecg = Input(shape=input_shape)

    x = Conv1D(64, 20, padding='same', activation='relu')(input_ecg)
    x = BatchNormalization()(x)

    x = resid_block(4, x)
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
    x = resid_block(4, x)

    decoder = Conv1D(6, 20, padding='same', activation='softmax')(x)

    model = Model(input_ecg, decoder)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def resid_block(size, resid, conv_size=20, filters_num=20):
    for i in np.arange(1, size):
        dil = [i ** 2]
        x = Conv1D(filters_num, conv_size, dilation_rate=dil, padding='same', activation='relu')(resid)
        x = BatchNormalization()(x)
        x = Conv1D(filters_num, conv_size, dilation_rate=dil, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        resid = Concatenate(axis=-1)([x, resid])
    return resid


if __name__ == "__main__":
    model = build_network((None, 1))
    model.summary()
    train_eval_holter(model, should_eval=False, should_load=True, MODEL_SAVE_PATH="models\\dense_detection_ma.h5")
