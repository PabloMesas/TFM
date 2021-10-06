import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ['CUDA_VISIBLE_DEVICES']='0' 
# import keras
from tensorflow import keras
import tensorflow as tf
# from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
# from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
# from keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.layers import BatchNormalization as BN
# from keras.layers import GaussianNoise as GN
# from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv3D, MaxPooling3D, MaxPool3D, GlobalAveragePooling3D, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
# from tensorflow.keras.models import Model
from vgg16_pseudo3D import brainVGG16_pseudo3D
from vox_cnn_pseudo3D import voxCNN_pseudo3D
from vox_cnn_pseudo3D_v2 import voxCNN_pseudo3D_V2
from vox_cnn_pseudo3D_v3 import voxCNN_pseudo3D_V3
from densenet_pseudo3D import denseNet121_pseudo3D
from denseNet_pseudoRGB import DenseNet_pseudoRGB
from tensorflow.keras import backend as K
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import Sequence
# from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.utils import plot_model, model_to_dot
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
# from skimage.transform import resize
# from sklearn import preprocessing
from sklearn.metrics import roc_curve,roc_auc_score
import pydot

from glob import glob

import SimpleITK as sitk

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

import datetime
x = datetime.datetime.today()

batch_size = 80
epochs = 120
shape=128
classes = ["AD", "CN"]
num_classes = len(classes) 
n_channels = 1
images_shape = (shape,shape,shape)

# **MODEL**
# model = brainVGG16_pseudo3D(input_shape=images_shape, n_classes=num_classes, pretrained=True, frozen=False,) # batch=16
model = denseNet121_pseudo3D(input_shape=images_shape, n_classes=num_classes, pretrained=False, frozen=False,) # batch=32 lr=0.000001
# model = DenseNet_pseudoRGB(input_shape=images_shape, classes=num_classes) # batch=32 lr=0.000001
# model = voxCNN_pseudo3D(input_shape=images_shape, n_classes=num_classes) # batch=32
# model = voxCNN_pseudo3D_V2(input_shape=images_shape, n_classes=num_classes) # batch=32
# model = voxCNN_pseudo3D_V3(input_shape=images_shape, n_classes=num_classes) # batch=32

# model.summary()

project_dir = "/home/pmeslaf/TFM/DATA/FIRST_VISIT_DATA_nougmented/"
from data_generator import DataGenerator

training_generator = DataGenerator(data_path=project_dir + '/Train/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   fourth_axis = False,
                                   shuffle=True)
valid_generator = DataGenerator(data_path=project_dir + '/Validation/',
                                   dim=images_shape,
                                   batch_size = 1,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=False,
                                   fourth_axis = False,
                                   shuffle=True)
test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape,
                                   batch_size = 1,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=True,
                                   fourth_axis = False,
                                   shuffle=True)

name_prefix = model.name + '_3x3_' + '_' + '-'.join(classes) + '_' + str(shape)
name_code = name_prefix + '_' + x.strftime("%d-%m-%Y_%H-%M")
name_epoch = model.name + '_E{epoch:02d}_' + '-'.join(classes) + '_' + str(shape) + '_' + x.strftime("%d-%m-%Y_%H-%M")

# graph_model = model_to_dot(model, show_shapes=True, show_layer_names=False,rankdir='LR')
# graph_model.write_svg(project_dir + name_prefix + '_model.svg')

# # Create a callback that saves the model's weights
checkpoint_path = project_dir +name_epoch+'.{val_accuracy:.4f}.m5'
callbacks_list = [
            # ReduceLROnPlateau(monitor='val_accuracy',
            #                   factor=0.5,
            #                   patience=30,
            #                   min_lr=0.000001,
            #                   verbose=1),
            ModelCheckpoint(filepath=checkpoint_path,
                            monitor='val_accuracy',
                            mode='max',
                            # monitor='val_loss',
                            # mode='min',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only = True),
            ModelCheckpoint(filepath=checkpoint_path,
                            monitor='val_accuracy',
                            mode='auto',
                            verbose=1,
                            period=10,
                            save_weights_only = True),
            CSVLogger( project_dir +name_code+'.log',
                      separator=',',
                      append=False)
    ]

opt = Adam(0.00001, beta_1=0.8, beta_2=0.8)

# Compile the model

# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_E67_AD-CN_128_04-10-2021_11-25.0.7333.m5')
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_E26_AD-CN_128_04-10-2021_11-25.0.7167.m5')
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_E66_AD-CN_128_23-09-2021_01-19.0.7812.m5')
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_E80_AD-CN_128_23-09-2021_21-00.0.7667.m5')
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_E02_MCI-CN_128_30-09-2021_18-40.0.7174.m5')
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_E36_AD-CN_128_05-10-2021_18-17.0.7667.m5')
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# K.set_value(model.optimizer.learning_rate, 0.00001)

# # Fit data to model
history = model.fit(x=training_generator,
                    epochs=epochs,
                    # initial_epoch=21,
                    verbose=1,
                    callbacks=callbacks_list,
                    use_multiprocessing=True,
                    workers=12,
                    validation_data=valid_generator)

# Test
# model.load_weights(project_dir + 'model_frozen_VoxResNet_ADMCI110_24-08-2021_00-57.03-0.748118.m5')
predictions = model.evaluate(test_generator,
                            use_multiprocessing=True,
                            workers=12)
print('Model Loss: %.4f' % (predictions[0]))
print('Model Accuracy: %.4f' % (predictions[1]))

if num_classes == 2:
    testy = test_generator.get_groung_truth_test()

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # predict probabilities
    lr_probs = model.predict(test_generator,
                            use_multiprocessing=True,
                            workers=12)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    testy = testy[:, 1]

    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name_prefix)
    # axis labels
    plt.xlabel('False Positive Rate (1 - Specifity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(project_dir + name_prefix + '.png')
    # plt.show()