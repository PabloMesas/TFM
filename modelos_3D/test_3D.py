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
from vox_cnn_v3 import VoxCNN_V3
from vox_cnn_v4 import VoxCNN_V4
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

batch_size = 10
epochs = 120
shape=128
classes = ["AD", "CN"]
num_classes = len(classes) 
n_channels = 1
images_shape = (shape,shape,int(shape), n_channels)

# **MODEL**
# model = VoxCNN(input_shape=images_shape, n_classes=num_classes) # batch=8
# model = VoxCNN_V2(input_shape=images_shape, n_classes=num_classes) # batch=8
# model = VoxCNN_V3(input_shape=images_shape, n_classes=num_classes) # batch=8
# model = VoxCNN_V4(input_shape=images_shape, n_classes=num_classes) # batch=8
# model = SimpleVoxCNN(input_shape=images_shape, n_classes=num_classes)
model = VoxResNet(input_shape=images_shape, n_classes=num_classes) # batch=4
# model = AllCNN(input_shape=images_shape, n_classes=num_classes)
# model = VoxInceptionCNN(input_shape=images_shape, n_classes=num_classes) # batch=16

# model.summary()

project_dir = "/home/pmeslaf/TFM/DATA/FIRST_VISIT_DATA_nougmented/"
from data_generator import DataGenerator

test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape[:-1],
                                   batch_size = 2,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=True,
                                   shuffle=True)

name_prefix = model.name + '_' + '-'.join(classes) + '_' + str(shape)
name_code = name_prefix + '_' + x.strftime("%d-%m-%Y_%H-%M")
name_epoch = model.name + '_E{epoch:02d}_' + '-'.join(classes) + '_' + str(shape) + '_' + x.strftime("%d-%m-%Y_%H-%M")

opt = Adam(0.00001, decay=1e-6)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Test
# model.load_weights(project_dir + 'VoxCNN_V4_E73_MCI-AD_128_10-11-2021_21-09.0.7150.m5')
# model.load_weights(project_dir + 'VoxCNN_V4_E45_MCI-AD_128_10-11-2021_21-09.0.7050.m5')

# AD-CN
# model.load_weights(project_dir + 'VoxCNN_V2_E44_AD-CN_128_26-10-2021_18-26.0.7633.m5')
# model.load_weights(project_dir + 'VoxCNN_V2_E52_AD-CN_128_02-11-2021_14-02.0.7733.m5')
# model.load_weights(project_dir + 'VoxCNN_V2_E37_AD-CN_128_03-11-2021_17-13.0.7833.m5')
# MCI-CN
# model.load_weights(project_dir + 'VoxCNN_V2_E03_MCI-CN_128_07-11-2021_18-38.0.7261.m5') #Acc 0.6200 ROC 0.690
# model.load_weights(project_dir + 'VoxCNN_V2_E10_MCI-CN_128_07-11-2021_18-38.0.6739.m5') #Acc 0.6800 ROC 0.744
# model.load_weights(project_dir + 'VoxCNN_V2_E20_MCI-CN_128_07-11-2021_18-38.0.6500.m5') #Acc 0.6300 ROC 0.745

# model.load_weights(project_dir + 'VoxCNN_V3_E10_AD-CN_128_04-11-2021_12-54.0.7767.m5')

# model.load_weights(project_dir + 'VoxCNN_V4_E14_AD-CN_128_04-11-2021_16-44.0.7600.m5') #Acc 0.8594 ROC 0.888
# model.load_weights(project_dir + 'VoxCNN_V4_E18_AD-CN_128_04-11-2021_16-44.0.7633.m5') #Acc 0.8750 ROC 0.870
# model.load_weights(project_dir + 'VoxCNN_V4_E19_AD-CN_128_04-11-2021_16-44.0.7900.m5') #Acc 0.8438 ROC 0.879
# model.load_weights(project_dir + 'VoxCNN_V4_E20_AD-CN_128_04-11-2021_16-44.0.7567.m5') #Acc 0.8906 ROC 0.881
# model.load_weights(project_dir + 'VoxCNN_V4_E23_AD-CN_128_04-11-2021_16-44.0.7933.m5') #Acc 0.8281 ROC 0.895
# model.load_weights(project_dir + 'VoxCNN_V4_E40_AD-CN_128_04-11-2021_16-44.0.8000.m5') #Acc 0.8906 ROC 0.903
# model.load_weights(project_dir + 'VoxCNN_V4_E50_AD-CN_128_04-11-2021_16-44.0.7667.m5') #Acc 0.9062 ROC 0.929
model.load_weights(project_dir + 'VoxCNN_V4_E90_AD-CN_128_04-11-2021_16-44.0.7700.m5') #Acc 0.9062 ROC 0.938
# model.load_weights(project_dir + 'VoxCNN_V4_E20_MCI-AD-CN_128_07-11-2021_20-22.0.4828.m5') #Acc 0.9062 ROC 0.938

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
    plt.savefig(project_dir + name_prefix + '.svg')
    # plt.show()