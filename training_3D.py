import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
# os.environ['CUDA_VISIBLE_DEVICES']='0' 
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
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
# from tensorflow.keras.models import Model
from vox_cnn import VoxCNN
from tensorflow.keras import backend as K
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import Sequence
# from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
# from skimage.transform import resize
# from sklearn import preprocessing

from glob import glob

import SimpleITK as sitk

# physical_devices = tf.config.list_physical_devices('GPU') 
# for gpu_instance in physical_devices: 
#     tf.config.experimental.set_memory_growth(gpu_instance, True)

batch_size = 4
epochs = 140
frozen_epochs=10
shape=128
n_slice_row = 4
classes = ["AD", "CN", "MCI"]
num_classes = len(classes) 
n_channels = 1
images_shape = (shape,shape,int(shape), n_channels)

import datetime
x = datetime.datetime.today()
name_code = 'TransferVGG19_' + classes[0] + classes[1] + str(shape) + '_' + x.strftime("%d-%m-%Y_%H-%M")


project_dir = "/home/pmeslaf/TFM/DATA/"
from data_generator import DataGenerator

training_generator = DataGenerator(data_path=project_dir + '/Train/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   num_classes=num_classes,
                                   shuffle=True,
                                   rotation=5)
valid_generator = DataGenerator(data_path=project_dir + '/Validation/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   num_classes=num_classes,
                                   shuffle=True)
test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   num_classes=num_classes,
                                   shuffle=True)

# # Create a callback that saves the model's weights
checkpoint_path = project_dir + 'model_frozen_'+name_code+'.{epoch:02d}-{val_loss:.6f}.m5'
callbacks_list = [
            # EarlyStopping(monitor='loss',
            #               min_delta=0,
            #               patience=2,
            #               verbose=1,
            #               mode='auto'),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              min_lr=0.000001,
                              verbose=1),
            ModelCheckpoint(filepath=checkpoint_path,
                            monitor='val_accuracy',
                            mode='max',
                            # monitor='val_loss',
                            # mode='min',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only = True),
            CSVLogger( project_dir + 'training_frozen_'+name_code+'.log',
                      separator=',',
                      append=False)
    ]



# **MODEL**

model = VoxCNN(input_shape=images_shape, n_classes=num_classes)

model.summary()

# opt = Adam(0.00001)


# # Compile the model
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# # Fit data to model frozen
# history = model.fit(x=training_generator,
#                     epochs=frozen_epochs,
#                     verbose=1,
#                     callbacks=callbacks_list,
#                     use_multiprocessing=True,
#                     workers=12,
#                     validation_data=valid_generator)


# #####DEFROST
# # # Create a callback that saves the model's weights
# checkpoint_path = project_dir + 'model_defrost_'+name_code+'.{epoch:02d}-{val_loss:.6f}.m5'
# callbacks_list = [
#             # EarlyStopping(monitor='loss',
#             #               min_delta=0,
#             #               patience=2,
#             #               verbose=1,
#             #               mode='auto'),
#             ReduceLROnPlateau(monitor='val_loss',
#                               factor=0.1,
#                               patience=3,
#                               min_lr=0.000001,
#                               verbose=1),
#             ModelCheckpoint(filepath=checkpoint_path,
#                             monitor='val_accuracy',
#                             mode='max',
#                             # monitor='val_loss',
#                             # mode='min',
#                             verbose=1,
#                             save_best_only=True,
#                             save_weights_only = True),
#             CSVLogger( project_dir + 'training_defrost_'+name_code+'.log',
#                       separator=',',
#                       append=False)
#     ]

# for layer in model.layers[:]:
#   layer.trainable = True
  
# # # Check the trainable status
# # for layer in model.layers:
# #   print(layer, layer.trainable)

# model.summary()

# opt2 = Adam(0.000001, decay=1e-6)

# # model.load_weights(project_dir + 'model_frozen_Inception_ADCN4x110_20-08-2021_18-20-14.14-0.576638.m5')
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt2,
#               metrics=['accuracy'])
# K.set_value(model.optimizer.learning_rate, 0.000001)

# history = model.fit(x=training_generator,
#                     epochs=epochs,
#                     initial_epoch=frozen_epochs,
#                     verbose=1,
#                     callbacks=callbacks_list,
#                     use_multiprocessing=True,
#                     workers=12,
#                     validation_data=valid_generator)