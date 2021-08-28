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
from vox_cnn import VoxCNN
from voxresnet import VoxResNet
from simple_3d_cnn import SimpleVoxCNN
from all_cnn import AllCNN
from vox_inception import VoxInceptionCNN
from vox_cnn_v2 import VoxCNN_V2
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

batch_size = 8
epochs = 80
shape=110
classes = ["MCI", "CN"]
num_classes = len(classes) 
n_channels = 1
images_shape = (shape,shape,int(shape), n_channels)

# **MODEL**
model = VoxCNN(input_shape=images_shape, n_classes=num_classes) # batch=8
# model = VoxCNN_V2(input_shape=images_shape, n_classes=num_classes) # batch=8
# model = SimpleVoxCNN(input_shape=images_shape, n_classes=num_classes)
# model = VoxResNet(input_shape=images_shape, n_classes=num_classes) # batch=4
# model = AllCNN(input_shape=images_shape, n_classes=num_classes)
# model = VoxInceptionCNN(input_shape=images_shape, n_classes=num_classes) # batch=16

model.summary()

project_dir = "/home/pmeslaf/TFM/DATA/FIRST_VISIT_DATA/"
from data_generator import DataGenerator

training_generator = DataGenerator(data_path=project_dir + '/Train/',
                                   dim=images_shape[:-1],
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                #    flip=True,
                                #    zoom=1.5,
                                #    rotation=10,
                                   shuffle=True)
valid_generator = DataGenerator(data_path=project_dir + '/Validation/',
                                   dim=images_shape[:-1],
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   shuffle=True)
test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape[:-1],
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   shuffle=True)

name_prefix = model.name + '_' + '-'.join(classes) + '_' + str(shape)
name_code = name_prefix + '_' + x.strftime("%d-%m-%Y_%H-%M")
name_epoch = model.name + '_E{epoch:02d}_' + '-'.join(classes) + '_' + str(shape) + '_' + x.strftime("%d-%m-%Y_%H-%M")

# graph_model = model_to_dot(model, show_shapes=True, show_layer_names=False,rankdir='LR')
# graph_model.write_svg(project_dir + name_prefix + '_model.svg')

# # Create a callback that saves the model's weights
checkpoint_path = project_dir +name_epoch+'.{val_accuracy:.4f}.m5'
callbacks_list = [
            # ReduceLROnPlateau(monitor='val_accuracy',
            #                   factor=0.1,
            #                   patience=5,
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

opt = Adam(0.000027, decay=1e-6)

# Compile the model

# model.load_weights(project_dir + 'VoxCNN_V2_E52_AD-CN_110_28-08-2021_11-19.0.8214.m5')
model.load_weights(project_dir + 'VoxCNN_E18_AD-CN_110_28-08-2021_00-45.0.7841.m5')
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# K.set_value(model.optimizer.learning_rate, 0.000001)

# Fit data to model
history = model.fit(x=training_generator,
                    epochs=epochs,
                    # initial_epoch=20,
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