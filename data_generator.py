import os
import random
from glob import glob
import concurrent.futures

import numpy as np
from sklearn import preprocessing
from scipy import ndimage
from skimage.transform import resize
import nibabel as nib

from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.utils import to_categorical

class DataGenerator(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                    data_path,
                    dim=(192,192,160),
                    batch_size = 1,
                    n_channels = 1,
                    classes = ["AD", "CN", "MCI"],
                    shuffle=True,
                    rotation=0):
        'Initialization'
        self.data_path = data_path
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.num_classes = len(classes)
        self.shuffle = shuffle
        self.rotation = rotation if rotation<=360 else rotation % 360
        self.classes = classes
        self.label_encoder = self.__set_label_encoder(self.classes)
        self.list_IDs, self.Y_labels = self.__get_index(data_path)
        self.on_epoch_end()
    
    def __get_index(self, data_path):
        files = [os.path.relpath(file_dir, data_path) for x in os.walk(data_path) for file_dir in glob(os.path.join(x[0], '*.nii.gz'))]
        x_index = []
        y_labels = []

        for f in files:
            f = f.split('/')
            if f[0] in self.classes:
                x_index.append(f[1])
                y_labels.append(f[0])

        y_labels = self.label_encoder.transform(y_labels)
        
        print(str(len(x_index)) + ' image(s) found in ' + data_path)

        return x_index, y_labels
    
    def __set_label_encoder(self, labels):
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        return le

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index): #Index denotes the current batch on an epoch
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]
        Batch_Y = [self.Y_labels[k] for k in indexes]
        
        # Generate data
        X = self.__data_generation(Batch_ids, Batch_Y)

        Batch_Y = to_categorical(Batch_Y, self.num_classes)

        return X, Batch_Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __crop_img(self, img):
        x0 = 0
        for slice in img:
            if np.all(slice < 0.000001):
                x0 +=1
            else:
                break
        
        x1 = 0
        for i in range(len(img), 0, -1):
            if np.all(img[i-1] < 0.00001):
                x1 +=1
            else:
                break
        
        y0 = 0
        for i in range(len(img[0])):
            if np.all(img[:,i] < 0.00001):
                y0 +=1
            else:
                break
        
        y1 = 0
        for i in range(len(img[0]), 0, -1):
            if np.all(img[:,i-1] < 0.00001):
                y1 +=1
            else:
                break

        z0 = 0
        for i in range(len(img[0,0])):
            if np.all(img[:,:,i] < 0.00001):
                z0 +=1
            else:
                break
        
        z1 = 0
        for i in range(len(img[0,0]), 0, -1):
            if np.all(img[:,:,i-1] < 0.00001):
                z1 +=1
            else:
                break

        return img[x0:-(1+x1),y0:-(1+y1),z0:-(1+z1)]
    
    def __load_data(self, img_path):
        # load nibabel Method
        img = nib.load(img_path).get_fdata()

        img = self.__crop_img(img)
        
        axes_list = [(0,1),(1,2),(0,2)]
        axes = random.choice(axes_list)
        if self.rotation > 0:
          angle = random.randint(-self.rotation, self.rotation)
          img = ndimage.rotate(img, angle, axes=axes, reshape=True)
        
        img = self.__crop_img(img)

        # # One more dimension for the channels
        img = np.expand_dims(img, axis=3)

        #NORMALIZATION
        # img = img/np.max(img) #We normalize between 0 an 1
        img = (img - np.mean(img)) / np.std(img) #whitening

        return img

    def __data_generation(self, Batch_ids, Batch_Y):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            # Generate data
            for c, i in enumerate(Batch_ids): #count, element
                case_path = os.path.join(self.data_path,
                                        self.label_encoder.inverse_transform([Batch_Y[c]])[0])
                img_path = os.path.join(case_path, i);

                img = self.__load_data(img_path=img_path)

                X[c,:,:,:,:] = resize(img, (self.dim[0], self.dim[1], self.dim[2], 1))
            

        return X
        
        # return X/np.max(X) #We normalize between 0 an 1
