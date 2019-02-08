from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, \
    Activation, MaxPooling2D, AveragePooling2D


def create_vgg_like_model(input_shape=(40, 256, 1), output_dim=256, filters=64, pool_shape=(1, 2), max_pool=True,
                          filter_shapes=[(1, 5), (1, 3)], dropout=0.3):

    model_name = 'vgg_like_in={}_out={}_filters={}_pool_shape={}_max={}_filter_shapes={}_dropout={}'\
        .format(input_shape, output_dim, filters, pool_shape, max_pool, filter_shapes, dropout)\
        .replace(',', '_').replace(' ', '_')

    visible = Input(shape=(input_shape[0], None, input_shape[2]))
    x = visible
    x = Conv2D(filters*2**0, filter_shapes[0], padding='same', activation='relu', name='Conv0')(x)
    x = BatchNormalization(name='BN0')(x)
    x = Conv2D(filters*2**0, filter_shapes[1], padding='same', activation='relu', name='Conv1')(x)
    x = BatchNormalization(name='BN1')(x)
    x = possible_pool(1, input_shape, pool_shape, max_pool, x)
    x = Dropout(dropout)(x)

    x = Conv2D(filters*2**1, filter_shapes[0], padding='same', activation='relu', name='Conv2')(x)
    x = BatchNormalization(name='BN2')(x)
    x = Conv2D(filters*2**1, filter_shapes[1], padding='same', activation='relu', name='Conv3')(x)
    x = BatchNormalization(name='BN3')(x)
    x = possible_pool(2, input_shape, pool_shape, max_pool, x)

    x = Dropout(dropout)(x)

    x = Conv2D(filters*2**2, filter_shapes[0], padding='same', activation='relu', name='Conv4')(x)
    x = BatchNormalization(name='BN4')(x)
    x = Conv2D(filters*2**2, filter_shapes[1], padding='same', activation='relu', name='Conv5')(x)
    x = BatchNormalization(name='BN5')(x)
    x = possible_pool(3, input_shape, pool_shape, max_pool, x)
    x = Dropout(dropout)(x)

    x = Conv2D(filters*2**2, filter_shapes[0], padding='same', activation='relu', name='Conv6')(x)
    x = BatchNormalization(name='BN6')(x)
    x = Conv2D(filters*2**2, filter_shapes[1], padding='same', activation='relu', name='Conv7')(x)
    x = BatchNormalization(name='BN7')(x)
    x = possible_pool(4, input_shape, pool_shape, max_pool, x)

    x = Dropout(dropout)(x)

    x = Conv2D(filters*2**3, filter_shapes[0], padding='same', activation='relu', name='Conv8')(x)
    x = BatchNormalization(name='BN8')(x)
    x = Conv2D(filters*2**3, filter_shapes[1], padding='same', activation='relu', name='Conv9')(x)
    x = BatchNormalization(name='BN9')(x)
    x = possible_pool(5, input_shape, pool_shape, max_pool, x)
    x = Dropout(dropout)(x)

    x = Conv2D(filters*2**3, filter_shapes[0], padding='same', activation='relu', name='Conv10')(x)
    x = BatchNormalization(name='BN10')(x)
    x = Conv2D(filters*2**3, filter_shapes[1], padding='same', activation='relu', name='Conv11')(x)
    x = BatchNormalization(name='BN11')(x)
    x = possible_pool(6, input_shape, pool_shape, max_pool, x)
    x = Dropout(dropout)(x)

    x = Conv2D(output_dim, (1, 1), padding='same', activation='relu', name='1x1')(x)
    x = GlobalAveragePooling2D()(x)

    x = Activation('softmax')(x)
    return Model(inputs=visible, outputs=x, name=model_name)


def possible_pool(layer, input_shape, pool_shape, max_pool, x):
    w = pool_shape[0]
    if input_shape[0] is not None and pool_shape[0] ** layer > input_shape[0]:
        w = 1

    h = pool_shape[1]
    if input_shape[1] is not None and pool_shape[1] ** layer > input_shape[1]:
        h = 1
    if max_pool:
        return MaxPooling2D((w, h), name='MaxPool2D' + str(layer))(x)
    else:
        return AveragePooling2D((w, h), name='AvgPool2D' + str(layer))(x)
