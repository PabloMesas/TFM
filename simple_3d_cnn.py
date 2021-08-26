from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

def SimpleVoxCNN(
    input_shape=(128,128,64,1),
    n_classes=3,
    classifier_activation='softmax'
):
    """Instantiates the VoxCNN architecture from:

        @inproceedings{goenka2021volumetric,
            title={Volumetric Convolutional Neural Network for Alzheimer Detection},
            author={Goenka, Nitika and Tiwari, Shamik},
            booktitle={2021 5th International Conference on Trends in Electronics and Informatics (ICOEI)},
            pages={1500--1505},
            year={2021},
            organization={IEEE}
        }

    """

    #Input
    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv3D(8, (2, 2, 2), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv3D(16, (2, 2, 2), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.Conv3D(32, (2, 2, 2), activation='relu', padding='same', name='block1_conv3')(x)
    x = layers.Conv3D(64, (2, 2, 2), activation='relu', padding='same', name='block1_conv4')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block1_pool')(x)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv3D(128, (2, 2, 2), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block2_pool')(x)
    x = layers.BatchNormalization()(x)

    # Block 3
    x = layers.Conv3D(256, (2, 2, 2), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.MaxPooling3D((2, 2, 2), name='block3_pool')(x)
    x = layers.BatchNormalization()(x)

    # Classification block
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, name='fc1', kernel_regularizer=regularizers.l1(0.001))(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.GaussianDropout(0.3)(x)

    x = layers.Dense(n_classes, activation=classifier_activation,
                        name='predictions')(x)

    # Create model.
    model = Model(img_input, x, name='SimpleVoxCNN')

    return model