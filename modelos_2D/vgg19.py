from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

def brainVGG19(
    input_shape=(128,128,3),
    frozen=False,
    pretrained=False,
    n_classes=3,
    classifier_activation='softmax'
):
    """Instantiates the VGG19 architecture pretrained with ImageNet
        
    """
    if pretrained:
        model1 = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        model1 = VGG19(include_top=False, weights=None, input_shape=input_shape)

    if frozen:
        for layer in model1.layers[:]:
            layer.trainable = False
    
    conv = model1.get_layer('block5_pool')

    # Classification block
    x = layers.Flatten(name='flatten')(conv.output)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)
    predictions = layers.Dense(n_classes, activation=classifier_activation, name='predictions')(x)

    model = Model(inputs=model1.input, outputs=predictions, name='vgg19')

    return model