from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, Activation


def create_shallow_key_model(input_shape=(168, 60, 1), output_dim=24, filters=4,
                             short_filter_length=3, long_filter_length=168, dropout=0.2):

    short_filter_shape = (short_filter_length, 1)
    long_filter_shape = (long_filter_length, 1)

    model_name = 'shallow_key_in={}_out={}_filters={}_short_shape={}_long_shape={}_dropout={}'\
        .format(input_shape, output_dim, filters, short_filter_shape, long_filter_shape, dropout) \
        .replace(',', '_').replace(' ', '_')

    visible = Input(shape=(input_shape[0], None, input_shape[2]))
    x = visible
    x = Conv2D(filters, short_filter_shape, padding='same', activation='relu', name='Conv0')(x)
    if dropout > 0.:
        x = Dropout(dropout)(x)
    x = AveragePooling2D(pool_size=(1, input_shape[1]), name='Avg0')(x)
    x = Conv2D(filters * 64, long_filter_shape, padding='same', activation='relu', name='Conv2')(x)
    if dropout > 0.:
        x = Dropout(dropout)(x)
    x = Conv2D(output_dim, (1, 1), padding='same', activation='relu', name='1x1')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return Model(inputs=visible, outputs=x, name=model_name)


def create_shallow_tempo_model(input_shape=(40, 256, 1), output_dim=256, filters=4,
                               short_filter_length=3, long_filter_length=256, dropout=0.2):
    short_filter_shape = (1, short_filter_length)
    long_filter_shape = (1, long_filter_length)

    model_name = 'shallow_tempo_in={}_out={}_filters={}_short_shape={}_long_shape={}_dropout={}'\
        .format(input_shape, output_dim, filters, short_filter_shape, long_filter_shape, dropout) \
        .replace(',', '_').replace(' ', '_')

    visible = Input(shape=(input_shape[0], None, input_shape[2]))
    x = visible
    x = Conv2D(filters, short_filter_shape, padding='same', activation='relu', name='Conv0')(x)
    if dropout > 0.:
        x = Dropout(dropout)(x)
    x = AveragePooling2D(pool_size=(input_shape[0], 1), name='Avg0')(x)
    x = Conv2D(filters * 64, long_filter_shape, padding='same', activation='relu', name='Conv2')(x)
    if dropout > 0.:
        x = Dropout(dropout)(x)
    x = Conv2D(output_dim, (1, 1), padding='same', activation='relu', name='1x1')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return Model(inputs=visible, outputs=x, name=model_name)
