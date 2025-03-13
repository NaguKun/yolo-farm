from keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, ZeroPadding2D, Flatten, Dropout
from keras.models import Model

def alexnet():
    """Define the AlexNet model."""
    input_1 = Input(shape=(227, 227, 3))

    conv_1 = Conv2D(96, 11, strides=(4, 4), activation='relu', name='conv_1')(input_1)
    pool_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool_1')(conv_1)
    norm_1 = BatchNormalization()(pool_1)
    padding_1 = ZeroPadding2D((2, 2))(norm_1)

    conv_2 = Conv2D(256, 5, activation='relu', name='conv_2')(padding_1)
    pool_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool_2')(conv_2)
    norm_2 = BatchNormalization()(pool_2)
    padding_2 = ZeroPadding2D((1, 1))(norm_2)

    conv_3 = Conv2D(384, 3, activation='relu', name='conv_3')(padding_2)
    padding_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Conv2D(384, 3, activation='relu', name='conv_4')(padding_3)
    padding_4 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Conv2D(256, 3, activation='relu', name='conv_5')(padding_4)
    pool_3 = MaxPooling2D((3, 3), strides=(3, 3), name='pool_3')(conv_5)

    dense_1 = Flatten(name="flatten")(pool_3)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_1)
    dense_2 = Dropout(0.5)(dense_2)
    dense_3 = Dense(10, activation='softmax', name='output')(dense_2)

    model = Model(inputs=input_1, outputs=dense_3)
    return model
