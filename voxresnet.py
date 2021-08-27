
"""ResNet models for Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.python.util.tf_export import keras_export


def ResNet(stack_fn,
            use_bias,
            model_name='resnet',
            input_shape=(128,128,128,1),
            n_classes=3,
            classifier_activation='softmax'):
    """Instantiates the VoxResNet from:

        @article{Korolev2017ResidualAP,
        title={Residual and plain convolutional neural networks for 3D brain MRI classification},
        author={Sergey Korolev and Amir Safiullin and M. Belyaev and Yulia Dodonova},
        journal={2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017)},
        year={2017},
        pages={835-838}
        }
    """

    img_input = layers.Input(shape=input_shape)

    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1 #3D images


    x = layers.Conv3D(32, 3, padding='same', use_bias=use_bias, name='conv1a_conv')(img_input)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1a_bn')(x)
    x = layers.Activation('relu', name='conv1a_relu')(x)

    x = layers.Conv3D(32, 3, padding='same', use_bias=use_bias, name='conv1b_conv')(x)

    x = stack_fn(x)

    # Classification block
    x = layers.MaxPooling3D(pool_size=(7, 7, 7), name='max_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)

    x = layers.Dense(n_classes, activation=classifier_activation,
                        name='predictions')(x)

    # Create model.
    model = training.Model(img_input, x, name=model_name)

    return model


def block(x, filters, kernel_size=3, name=None):
    """A residual block.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
        name: string, block label.

    Returns:
        Output tensor for the residual block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1

    shortcut = x

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv3D(filters, kernel_size, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv3D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)


    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack(x, filters, blocks, name=None):
    """A set of stacked residual blocks.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        name: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    x = layers.Conv3D(filters, 3, padding='same', strides=2, name=name + '_0_conv')(x)

    x = block(x, filters, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block(x, filters, name=name + '_block' + str(i))
    return x


def VoxResNet(input_shape=None,
                n_classes=3):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack(x, filters=64, blocks=2, name='conv2')
        x = stack(x, filters=64, blocks=2, name='conv3')
        return stack(x, filters=128, blocks=2, name='conv4')

    return ResNet(stack_fn, True, model_name='VoxResNet',
                                input_shape=input_shape, n_classes=n_classes)

