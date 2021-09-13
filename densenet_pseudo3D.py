from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121

def denseNet121_pseudo3D(
    input_shape=(128,128,128),
    frozen=False,
    pretrained=False,
    n_classes=3,
    classifier_activation='softmax'
):
    """Instantiates the VGG16 architecture pretrained with ImageNet with an 3D input 
        
    """

    #Input
    img_input = layers.Input(shape=input_shape)
    # Block 1 - PseudoRGB with 3  filters
    input_psudoRGB = layers.Conv2D(3, (3, 3), activation='relu', padding='same', name='block1_pseudoRGB')(img_input)
    #TODO: Cambiar tamaño del filtro 3x3 o 1x1

    if pretrained:
        model1 = DenseNet121(include_top=False, weights="imagenet", input_shape=(input_shape[0], input_shape[1], 3))
    else:
        model1 = DenseNet121(include_top=False, weights=None, input_shape=(input_shape[0], input_shape[1], 3))

    if frozen:
        for layer in model1.layers[:]:
            layer.trainable = False

    model1.summary()
    conv = model1.get_layer('relu')
    # Classification block
    # x = layers.Flatten(name='flatten')(conv.output)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(conv.output)
    # x = layers.Dense(128, activation='relu', name='fc1')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # x = layers.Dense(64, activation='relu', name='fc2')(x)
    predictions = layers.Dense(n_classes, activation=classifier_activation, name='predictions')(x)

    model = Model(inputs=model1.inputs, outputs=predictions)

    
    #Imput part
    x = model(input_psudoRGB)
    model = Model(img_input, x, name='denseNet121_pseudoRGB')
    # model.summary()

    return model