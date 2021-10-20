from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

def voxCNN_pseudo3D(
    input_shape=(128,128,128),
    n_classes=3,
    classifier_activation='softmax'
):
    """2D VoxCNN architecture with an 3D input 
        
    """

    #Input
    img_input = layers.Input(shape=input_shape)
    x = layers.experimental.preprocessing.RandomFlip("vertical")(img_input)
    x = layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='constant')(x) #nearest
    x = layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest')(x)
    x = layers.experimental.preprocessing.RandomZoom(height_factor=0.3, fill_mode='constant')(x)

    # Block 1 - PseudoRGB with 3  filters
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_64')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_32')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_16')(x)
    input_psudoRGB = layers.Conv2D(3, (3, 3), activation='relu', padding='same', name='block0_pseudoRGB')(x)
    #TODO: Cambiar tamaño del filtro 3x3 o 1x1

   # Block 1
    x = layers.Conv2D(
        8, (3, 3), activation='relu', padding='same', name='block1_conv1')(
            input_psudoRGB)
    x = layers.Conv2D(
        8, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        16, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        16, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='block4_pool')(x)

    # # Block 5
    # x = layers.Conv2D(
    #     512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = layers.Conv2D(
    #     512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = layers.Conv2D(
    #     512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = layers.MaxPooling2D((2, 2), name='block5_pool')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)

    x = layers.Dense(n_classes, activation=classifier_activation,
                        name='predictions')(x)

    # Create model.
    model = Model(img_input, x, name='VoxCNN_pseudoRGB')

    return model