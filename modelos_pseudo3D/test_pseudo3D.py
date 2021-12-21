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

batch_size = 16
epochs = 120
shape=128
classes = ["AD", "CN"]
num_classes = len(classes) 
n_channels = 1
images_shape = (shape,shape,shape)

# **MODEL**
model = brainVGG16_pseudo3D(input_shape=images_shape, n_classes=num_classes, pretrained=True, frozen=False,) # batch=16
# model = denseNet121_pseudo3D(input_shape=images_shape, n_classes=num_classes, pretrained=True, frozen=False,) # batch=32 lr=0.000001
# model = voxCNN_pseudo3D(input_shape=images_shape, n_classes=num_classes) # batch=32

# model = DenseNet_pseudoRGB(input_shape=images_shape, classes=num_classes) # batch=32 lr=0.000001
# model = voxCNN_pseudo3D_V2(input_shape=images_shape, n_classes=num_classes) # batch=32
# model = voxCNN_pseudo3D_V3(input_shape=images_shape, n_classes=num_classes) # batch=32

# model.summary()

project_dir = "/home/pmeslaf/TFM/DATA/FIRST_VISIT_DATA_nougmented/"
from data_generator_ps3D import DataGenerator

test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape,
                                   batch_size = 10,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=True,
                                   shuffle=True)

name_prefix = model.name + '_' + '-'.join(classes) + '_' + str(shape)
name_code = name_prefix + '_' + x.strftime("%d-%m-%Y_%H-%M")
name_epoch = model.name + '_' + x.strftime("%d-%m-%Y_%H-%M") + '_E{epoch:02d}_' + '-'.join(classes) + '_' + str(shape) 

opt = Adam(0.000027, decay=1e-6)
# opt = RMSprop(0.00001, decay=1e-6)
# opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True) #VoxResnet

# Compile the model

#AD - CN
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_12-11-2021_17-16_E117_AD-CN_128.0.6458.m5') #ROC 0.618 ACC. 0.6406
model.load_weights(project_dir + 'VGG16_pseudoRGB_12-11-2021_17-14_E114_AD-CN_128.0.7083.m5') #ROC 0.822 ACC. 0.8218

#MCI - AD
# model.load_weights(project_dir + 'VGG16_pseudoRGB_14-11-2021_23-35_E03_MCI-AD_128.0.7135.m5') #ROC 0.590 ACC. 0.7159
# model.load_weights(project_dir + 'VGG16_pseudoRGB_14-11-2021_23-35_E40_MCI-AD_128.0.7083.m5') #ROC 0.670 ACC. 0.7159
# model.load_weights(project_dir + 'VGG16_pseudoRGB_15-11-2021_12-44_E80_MCI-AD_128.0.7100.m5') #ROC 0.685 ACC. 0.7159
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_15-11-2021_12-46_E80_MCI-AD_128.0.7031.m5') #ROC 0.575 ACC. 0.6818
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_14-11-2021_23-35_E27_MCI-AD_128.0.7135.m5') #ROC 0.475 ACC. 0.6705
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_14-11-2021_19-40_E58_MCI-AD_128.0.6927.m5') #ROC 0.561 ACC. 0.6932
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_15-11-2021_10-34_E04_MCI-AD_128.0.7188.m5') #ROC 0.397 ACC. 0.7045
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_15-11-2021_10-34_E20_MCI-AD_128.0.6875.m5') #ROC 0.469 ACC. 0.7045
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_15-11-2021_10-34_E02_MCI-AD_128.0.7083.m5') #ROC 0.406 ACC. 0.7045
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_15-11-2021_10-34_E50_MCI-AD_128.0.6771.m5') #ROC 0.446 ACC. 0.6932

#MCI - CN
# model.load_weights(project_dir + 'VGG16_pseudoRGB_18-11-2021_00-33_E52_MCI-CN_128.0.6161.m5') #ROC 0.506 ACC. 0.6200
# model.load_weights(project_dir + 'VGG16_pseudoRGB_18-11-2021_00-33_E70_MCI-CN_128.0.6295.m5') #ROC 0.576 ACC. 0.5900
# model.load_weights(project_dir + 'VGG16_pseudoRGB_18-11-2021_00-33_E75_MCI-CN_128.0.6652.m5') #ROC 0.573 ACC. 0.5900
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_15-11-2021_12-46_E80_MCI-AD_128.0.7031.m5') #ROC 0.575 ACC. 0.6818
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_14-11-2021_23-35_E27_MCI-AD_128.0.7135.m5') #ROC 0.475 ACC. 0.6705
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_14-11-2021_19-40_E58_MCI-AD_128.0.6927.m5') #ROC 0.561 ACC. 0.6932
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_19-11-2021_13-30_E84_MCI-CN_128.0.6696.m5') #ROC 0.578 ACC. 0.6000
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_18-11-2021_09-53_E68_MCI-CN_128.0.6696.m5') #ROC 0.602 ACC. 0.6000
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_18-11-2021_09-53_E43_MCI-CN_128.0.6562.m5') #ROC 0.614 ACC. 0.6200

#MCI - AD - CN
# model.load_weights(project_dir + 'VGG16_pseudoRGB_18-11-2021_00-33_E52_MCI-CN_128.0.6161.m5') #ROC 0.506 ACC. 0.6200
# model.load_weights(project_dir + 'VGG16_pseudoRGB_18-11-2021_00-33_E70_MCI-CN_128.0.6295.m5') #ROC 0.576 ACC. 0.5900
# model.load_weights(project_dir + 'VGG16_pseudoRGB_18-11-2021_00-33_E75_MCI-CN_128.0.6652.m5') #ROC 0.573 ACC. 0.5900
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_18-11-2021_12-12_E62_MCI-CN-AD_128.0.5471.m5') #ACC. 0.4524
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_18-11-2021_12-12_E68_MCI-CN-AD_128.0.5580.m5') #ACC. 0.4762
# model.load_weights(project_dir + 'denseNet121_pseudoRGB_18-11-2021_12-12_E14_MCI-CN-AD_128.0.5181.m5') #ACC. 0.4683
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_16-11-2021_18-32_E110_MCI-CN-AD_128.0.5486.m5') #ACC. 0.5238
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_16-11-2021_18-32_E56_MCI-CN-AD_128.0.5278.m5') #ACC. 0.4921
# model.load_weights(project_dir + 'VoxCNN_pseudoRGB_16-11-2021_18-32_E51_MCI-CN-AD_128.0.5243.m5') #ACC. 0.4365



model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Test
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