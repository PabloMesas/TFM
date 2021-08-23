from tensorflow.keras import layers
from tensorflow.keras.models import Model

def VoxCNN(
    input_shape=(128,128,128,1),
    n_classes=3,
    classifier_activation='softmax'
):
    """Instantiates the VoxCNN architecture from:

        @article{Korolev2017ResidualAP,
        title={Residual and plain convolutional neural networks for 3D brain MRI classification},
        author={Sergey Korolev and Amir Safiullin and M. Belyaev and Yulia Dodonova},
        journal={2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017)},
        year={2017},
        pages={835-838}
        }

    """

    #Input
    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv3D(
        8, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(
            img_input)
    x = layers.Conv3D(
        8, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv3D(
        16, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv3D(
        16, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv3D(
        32, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv3D(
        32, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv3D(
        32, (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv3D(
        64, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv3D(
        64, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv3D(
        64, (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block4_pool')(x)

    # # Block 5
    # x = layers.Conv3D(
    #     512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = layers.Conv3D(
    #     512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = layers.Conv3D(
    #     512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = layers.MaxPooling3D((2, 2, 2), name='block5_pool')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)

    x = layers.Dense(n_classes, activation=classifier_activation,
                        name='predictions')(x)

    # Create model.
    model = Model(img_input, x, name='VoxCNN')

    return model