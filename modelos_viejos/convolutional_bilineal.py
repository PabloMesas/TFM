import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ['CUDA_VISIBLE_DEVICES']='0' 
# import keras
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
# from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
# from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization as BN
# from keras.layers import GaussianNoise as GN
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv3D, MaxPooling3D, MaxPool3D, GlobalAveragePooling3D, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
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

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

batch_size = 4
epochs = 100
frozen_epochs=30
shape=110
n_slice_row = 7
classes = ["AD", "CN", "MCI"]
num_classes = len(classes) 
n_channels = 1
images_shape = (shape*n_slice_row, shape*n_slice_row, n_channels*3)


project_dir = "/home/pmeslaf/TFM/DATA/"
from data_generator_2d import DataGenerator2D

training_generator = DataGenerator2D(data_path=project_dir + '/Train/',
                                   slice_size_dim=shape,
                                   n_slice_row=n_slice_row,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   shuffle=True,
                                   RGB=True,
                                   rotation=300)
valid_generator = DataGenerator2D(data_path=project_dir + '/Validation/',
                                   slice_size_dim=shape,
                                   n_slice_row=n_slice_row,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   RGB=True,
                                   shuffle=True)
test_generator = DataGenerator2D(data_path=project_dir + '/Test/',
                                   slice_size_dim=shape,
                                   n_slice_row=n_slice_row,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   classes = classes,
                                   RGB=True,
                                   shuffle=True)

# # Create a callback that saves the model's weights
checkpoint_path = project_dir + 'model_frozen.{epoch:02d}-{val_loss:.6f}.m5'
callbacks_list = [
            # EarlyStopping(monitor='loss',
            #               min_delta=0,
            #               patience=2,
            #               verbose=1,
            #               mode='auto'),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=10,
                              min_lr=0.000001,
                              verbose=1),
            ModelCheckpoint(filepath=checkpoint_path,
                            # monitor='val_accuracy',
                            # mode='max',
                            monitor='val_loss',
                            mode='min',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only = True),
            CSVLogger( project_dir + 'training_frozen.log',
                      separator=',',
                      append=False)
    ]



# **MODEL**

model1 = VGG16(include_top=False, weights="imagenet", input_shape=images_shape)
# model1 = InceptionV3(input_shape=x_train_shape, weights='imagenet', include_top=False)
for layer in model1.layers[:]:
  layer.trainable = False
  
# Check the trainable status
for layer in model1.layers:
  print(layer, layer.trainable, layer.name)

model1.summary()

def outer_product(x):
  phi_I = tf.einsum('ijkm,ijkn->imn',x[0],x[1])		# Einstein Notation  [batch,31,31,depth] x [batch,31,31,depth] -> [batch,depth,depth]
  phi_I = tf.reshape(phi_I,[-1,512*512])	        # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
  phi_I = tf.divide(phi_I,15*15)								  # Divide by feature map size [sizexsize]

  y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))		# Take signed square root of phi_I
  z_l2 = tf.nn.l2_normalize(y_ssqrt)								              # Apply l2 normalization
  return z_l2


conv=model1.get_layer('block5_conv3') 
d1=Dropout(0.5)(conv.output)   ## Why??
d2=Dropout(0.5)(conv.output)   ## Why??

x = Lambda(outer_product, name='outer_product')([d1,d2])

predictions=Dense(num_classes, activation='softmax', name='predictions')(x)

model = Model(inputs=model1.input, outputs=predictions)
model.summary()

# opt = Adam(0.01)


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


#####DEFROST
# # Create a callback that saves the model's weights
checkpoint_path = project_dir + 'model_defrost.{epoch:02d}-{val_loss:.6f}.m5'
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
                            # monitor='val_accuracy',
                            # mode='max',
                            monitor='val_loss',
                            mode='min',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only = True),
            CSVLogger( project_dir + 'training_defrost.log',
                      separator=',',
                      append=False)
    ]

for layer in model.layers[:]:
  layer.trainable = True
  
# Check the trainable status
for layer in model.layers:
  print(layer, layer.trainable)

model.summary()

opt = Adam(0.0000001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.load_weights(project_dir + 'model_frozen.05-1.020486.m5')
K.set_value(model.optimizer.learning_rate, 0.000001)

history = model.fit(x=training_generator,
                    epochs=epochs,
                    initial_epoch=frozen_epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    use_multiprocessing=True,
                    workers=12,
                    validation_data=valid_generator)