from keras.layers import *
from keras.models import Model

from evaluation import train_eval_holter


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


def unet_1d(
        input_shape,
        num_classes=1,
        use_batch_norm=True,
        use_dropout_on_upsampling=False,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=3,
        output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

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

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


if __name__ == "__main__":
    model = unet_1d(
        input_shape=(None, 1),
        use_batch_norm=True,
        num_classes=6,
        filters=64,
        dropout=0.5,
        output_activation='softmax')
    model.summary()
    train_eval_holter(model, should_eval=True, should_load=True, MODEL_SAVE_PATH = "models\\unet_detection_em.h5")