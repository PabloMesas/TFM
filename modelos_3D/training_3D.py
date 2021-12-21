import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
# os.environ['CUDA_VISIBLE_DEVICES']='1' 
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

from tensorflow.keras import layers
from tensorflow.keras.models import Model

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

import datetime
x = datetime.datetime.today()

batch_size = 4
epochs = 120
shape=100
classes = ["MCI", "AD"]
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

model.summary()

project_dir = "/home/pmeslaf/TFM/DATA/FIRST_VISIT_DATA_nougmented/"
from data_generator import DataGenerator

training_generator = DataGenerator(data_path=project_dir + '/Train/',
                                   dim=images_shape[:-1],
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=False,
                                    shuffle=True,
                                    flip=True,
                                    zoom=0.3,
                                    rotation=40)
valid_generator = DataGenerator(data_path=project_dir + '/Validation/',
                                   dim=images_shape[:-1],
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=False,
                                    shuffle=True,
                                    flip=True,
                                    zoom=0.3,
                                    rotation=40)
test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape[:-1],
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   test=True,
                                   shuffle=True)

name_prefix = model.name + '_' + '-'.join(classes) + '_' + str(shape)
name_code = name_prefix + '_' + x.strftime("%d-%m-%Y_%H-%M")
name_epoch = model.name + '_' + x.strftime("%d-%m-%Y_%H-%M") + '_E{epoch:02d}_' + '-'.join(classes) + '_' + str(shape) 

# graph_model = model_to_dot(model, show_shapes=True, show_layer_names=False,rankdir='LR')
# graph_model.write_svg(project_dir + name_prefix + '_model.svg')

# # Create a callback that saves the model's weights
checkpoint_path = project_dir +name_epoch+'.{val_accuracy:.4f}.m5'
callbacks_list = [
            # ReduceLROnPlateau(monitor='val_accuracy',
            #                   factor=0.1,
            #                   patience=15,
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


# opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True) #VoxResnet
opt = Adam(0.000027, decay=1e-6)
# opt = RMSprop(0.1)

# Compile the model

model.load_weights(project_dir + 'VoxCNN_V4_E30_MCI-AD_128_06-11-2021_10-18.0.6150.m5')
# model.load_weights(project_dir + 'VoxCNN_E27_AD-CN_128_26-10-2021_18-27.0.7867.m5')
# model.load_weights(project_dir + 'VoxCNN_V2_E44_AD-CN_128_26-10-2021_18-26.0.7633.m5')
# model.load_weights(project_dir + 'VoxCNN_V2_E10_MCI-CN_128_07-11-2021_18-38.0.6739.m5') #Acc 0.6800 ROC 0.744

# model.load_weights(project_dir + 'VoxCNN_V4_E20_AD-CN_128_04-11-2021_16-44.0.7567.m5') #Acc 0.8906 ROC 0.881
# model.load_weights(project_dir + 'VoxCNN_V4_E10_MCI-AD-CN_128_08-11-2021_12-58.0.4448.m5') #Acc 0.9062 ROC 0.938
# VoxCNN_V4_E10_MCI-AD-CN_128_08-11-2021_12-58.0.4448.m5
# dense_2=model.get_layer('fc2') 

# predictions=layers.Dense(num_classes, activation='softmax', name='predictions')(dense_2.output)

# model.load_weights(project_dir + 'VoxResNet_E12_AD-CN_100_06-11-2021_10-26.0.7568.m5') #Acc 0.5938 ROC 0.602

# model.load_weights(project_dir + 'VoxResNet_E112_AD-CN_100_06-11-2021_13-05.0.8176.m5') #Acc 0.8594 ROC 0.909

# dense_2=model.get_layer('fc1') 

# predictions=layers.Dense(num_classes, activation='softmax', name='predictions')(dense_2.output)

# model = Model(inputs=model.input, outputs=predictions)
# model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# K.set_value(model.optimizer.learning_rate, 0.0001)

# Fit data to model
history = model.fit(x=training_generator,
                    epochs=epochs,
                    # initial_epoch=12,
                    verbose=1,
                    callbacks=callbacks_list,
                    use_multiprocessing=True,
                    workers=3,
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
#     # plt.show()

# def get_model_memory_usage(batch_size, model):
#     import numpy as np
#     try:
#         from keras import backend as K
#     except:
#         from tensorflow.keras import backend as K

#     shapes_mem_count = 0
#     internal_model_mem_count = 0
#     for l in model.layers:
#         layer_type = l.__class__.__name__
#         if layer_type == 'Model':
#             internal_model_mem_count += get_model_memory_usage(batch_size, l)
#         single_layer_mem = 1
#         out_shape = l.output_shape
#         if type(out_shape) is list:
#             out_shape = out_shape[0]
#         for s in out_shape:
#             if s is None:
#                 continue
#             single_layer_mem *= s
#         shapes_mem_count += single_layer_mem

#     trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
#     non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

#     number_size = 4.0
#     if K.floatx() == 'float16':
#         number_size = 2.0
#     if K.floatx() == 'float64':
#         number_size = 8.0

#     total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
#     gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
#     return gbytes


# print(model.name)
# print(get_model_memory_usage(batch_size, model))