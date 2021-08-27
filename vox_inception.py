from tensorflow.keras import layers
from tensorflow.keras.models import Model

def VoxInceptionCNN(
    input_shape=(128,128,128,1),
    n_classes=3,
    classifier_activation='softmax'
):
    """Instantiates the Inception3D architecture from:
        Me
    """

    #Input
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv3D(32, (7, 7, 7), strides=2, padding='same', activation='relu', name='block1_conv1')(img_input)
    x = layers.MaxPooling3D((3, 3, 3), strides=2, name='max_pool_1')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(32, (1, 1, 1), strides=1, padding='valid', activation='relu', name='block1_conv2')(img_input)

    x = layers.Conv3D(32, (3, 3, 3), strides=2, padding='same', activation='relu', name='block1_conv2')(img_input)
    x = layers.MaxPool3D((3, 3, 3), strides=2, name='max_pool_2')(x)
    x = layers.BatchNormalization()(x)

    x = inception_module(x,
                        k=8,
                        name='inception_1a')

    x = inception_module(x,
                        k=16,
                        name='inception_1b')
    
    x = layers.MaxPool3D((3, 3, 3), padding='same', strides=2, name='max_pool_3')(x)

    x = inception_module(x,
                        k=32,
                        name='inception_2a')

    x = inception_module(x,
                        k=64,
                        name='inception_2b')
    
    x = layers.MaxPool3D((3, 3, 3), padding='same', strides=2, name='max_pool_4')(x)

    x = inception_module(x,
                        k=128,
                        name='inception_3a')

    x = inception_module(x,
                        k=256,
                        name='inception_3b')
    
    x = layers.MaxPool3D((3, 3, 3), padding='same', strides=2, name='max_pool_5')(x)

    # Classification block
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(n_classes, activation=classifier_activation, name='predictions')(x)

    # Create model.
    model = Model(img_input, x, name='VoxInception')

    return model


def inception_module(x,
                    k,
                    name=None):
    path1 = layers.Conv3D(int(k/2), (1, 1, 1), padding='same', activation='relu')(x)

    path2 = layers.Conv3D(int(k*3/4), (1, 1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv3D(k, (3, 3, 3), padding='same', activation='relu')(path2)

    path3 = layers.Conv3D(int(k/4), (1, 1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv3D(int(k/2), (5, 5, 5), padding='same', activation='relu')(path3)

    path4 = layers.MaxPool3D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv3D(int(k/4), (1, 1, 1), padding='same', activation='relu')(path4)
    
    return layers.concatenate([path1, path2, path3, path4], axis=3, name=name)