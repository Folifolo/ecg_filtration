from keras.layers import *
from keras.models import Model

from evaluation import load_split, train_eval


def conv1d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=10,
        activation='relu',
        kernel_initializer='he_normal',
        padding='same'):
    c = Conv1D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv1D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def build_unet_1d(
        input_shape,
        num_classes=6,
        use_batch_norm=True,
        use_dropout_on_upsampling=False,
        dropout=0.5,
        dropout_change_per_layer=0.0,
        filters=64,
        num_layers=3,
        output_activation='softmax'):  # 'sigmoid' or 'softmax'

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv1d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling1D(2)(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv1d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = UpSampling1D(2)(x)
        x = concatenate([x, conv])
        x = conv1d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    outputs = Conv1D(num_classes, 1, activation=output_activation)(x)

    unet_model = Model(inputs=[inputs], outputs=[outputs])
    unet_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return unet_model


if __name__ == "__main__":
    MODEL_PATH = "unet_detection"

    model = build_unet_1d((None, 1))
    # model = load_model(MODEL_PATH)
    model.summary()

    X = load_split()
    train_eval(model, X, only_eval=True, save_path=MODEL_PATH, size=2048, epochs=150)
