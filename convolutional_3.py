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
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, MaxPool3D, GlobalAveragePooling3D, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
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
import pandas as pd
import SimpleITK as sitk

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

batch_size = 12
epochs = 300
# frozen_epochs = 100
num_classes = 2
shape=110
images_shape = (shape,shape,int(shape))
n_channels = 1


project_dir = "/home/pmeslaf/TFM/DATA/"
from data_generator import DataGenerator

training_generator = DataGenerator(data_path=project_dir + '/Train/',
                                   dim=images_shape,
                                   batch_size = batch_size,
                                   n_channels = n_channels,
                                   num_classes=num_classes,
                                   shuffle=True,
                                   rotation=360)
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
checkpoint_path = project_dir + 'model_.{epoch:02d}-{val_loss:.6f}.m5'
callbacks_list = [
            # EarlyStopping(monitor='loss',
            #               min_delta=0,
            #               patience=2,
            #               verbose=1,
            #               mode='auto'),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=3,
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
            CSVLogger( project_dir + 'training.log',
                      separator=',',
                      append=False)
    ]



# **MODEL**

def CBGN(model,filters,lname,ishape=0):
  if (ishape!=0):
    model.add(Conv3D(filters=filters, kernel_size=3, activation="relu",
                 input_shape=ishape))
  else:
    model.add(Conv3D(filters=filters, kernel_size=3, activation="relu"))

  # model.add(MaxPool3D(pool_size=2,name=lname))
  
  return model


model = Sequential()

model=CBGN(model,16,'conv_model_1',(images_shape[0], images_shape[1], images_shape[2], 1))
model=CBGN(model,16,'conv_model_2')
model.add(MaxPool3D(pool_size=2))
model=CBGN(model,32,'conv_model_3')
model=CBGN(model,32,'conv_model_4')
model.add(MaxPool3D(pool_size=2))
model=CBGN(model,64,'conv_model_5')
model=CBGN(model,64,'conv_model_6')
model=CBGN(model,64,'conv_model_7')
model.add(MaxPool3D(pool_size=2))
model=CBGN(model,128,'conv_model_8')
model=CBGN(model,128,'conv_model_9')
model=CBGN(model,128,'conv_model_10')
model.add(MaxPool3D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128))
model.add(BN())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation="relu"))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
# model.add(Activation('sigmoid'))

model.summary()

opt = Adam(0.00001, decay=1e-6)


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Fit data to model
# model.load_weights(project_dir + 'model_.197-0.546914.m5')
history = model.fit(training_generator,
                    epochs=epochs,
                    verbose=1,
                    # initial_epoch=100,
                    callbacks=callbacks_list,
                    use_multiprocessing=True,
                    workers=12,
                    validation_data=valid_generator)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(project_dir + 'evolution_training.png')
# plt.show()


# history = pd.read_csv(project_dir + 'training.log', sep=',', engine='python')
# plt.plot(history['accuracy'])
# plt.plot(history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig(project_dir + 'evolution_training.png')


predictions = model.evaluate(test_generator)

print('Test loss:', predictions[0])
print('Test accuracy:', predictions[1])