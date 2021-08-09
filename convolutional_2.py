import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
# os.environ['CUDA_VISIBLE_DEVICES']='1' 
# import keras
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
# from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
# from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization as BN
# from keras.layers import GaussianNoise as GN
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, MaxPool3D, GlobalAveragePooling3D, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
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

batch_size = 6
epochs = 100
# frozen_epochs = 100
num_classes = 3
images_shape = (100,120,70)
n_channels = 1


project_dir = "/home/pmeslaf/DATA/"
from data_generator import DataGenerator

training_generator = DataGenerator(data_path=project_dir + '/Train/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   num_classes=num_classes,
                                   shuffle=True,
                                   rotation=60)
# valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(data_path=project_dir + '/Test/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   num_classes=num_classes,
                                   shuffle=False)

# **MODEL**

## DEF A BLOCK CONV + BN + GN + CONV + BN + GN + MAXPOOL 
def CBGN(model,filters,lname,ishape=0):
  if (ishape!=0):
    model.add(Conv3D(filters=filters, kernel_size=3, activation="relu",
                 input_shape=ishape))
  else:
    model.add(Conv3D(filters=filters, kernel_size=3, activation="relu"))

  model.add(MaxPool3D(pool_size=2,name=lname))
  
  return model


model = Sequential()

model=CBGN(model,32,'conv_model_1',(images_shape[0], images_shape[1], images_shape[2], 1))
model=CBGN(model,64,'conv_model_2')
model=CBGN(model,128,'conv_modeL_3')
model=CBGN(model,256,'conv_modeL_4')
# model=CBGN(model,512,'conv_model_5')

# model.add(Flatten())
model.add(GlobalAveragePooling3D())
model.add(Dense(units=512, activation="relu"))
model.add(BN())
# model.add(GN(0.3))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
# model.add(Activation('sigmoid'))

model.summary()

opt = Adam(1, decay=1e-6)


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(x=training_generator,
                    epochs=epochs,
                    verbose=1,
                    validation_data = test_generator)
