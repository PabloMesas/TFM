import os
import random
from glob import glob
import concurrent.futures

import numpy as np
from scipy import ndimage
from skimage.transform import resize
import nibabel as nib

class DataAugmentation():
    'Generates and saves data'
    def __init__(self,
                    data_path,
                    classes = ["AD", "CN", "MCI"],
                    flip=False,
                    zoom=0.0):
        'Initialization'
        self.data_path = data_path
        self.flip = flip
        self.zoom=zoom
        self.classes = classes

        self.list_IDs, self.Y_labels = self.__get_index(data_path)

    
    def __get_index(self, data_path):
        files = [os.path.relpath(file_dir, data_path) for x in os.walk(data_path) for file_dir in glob(os.path.join(x[0], '*.nii.gz'))]
        x_index = []
        y_labels = []

        for f in files:
            f = f.split('/')
            if f[0] in self.classes:
                x_index.append(f[1])
                y_labels.append(f[0])

        # y_labels = self.label_encoder.transform(y_labels)
        
        print(str(len(x_index)) + ' image(s) found in ' + data_path)

        return x_index, y_labels
    
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
    
    def make_zoom(self, img):
        zoom = round(random.uniform(1.05, self.zoom), 2)
        if zoom > 1.0:
            original_shape = img.shape
            img = ndimage.zoom(img, zoom)
            zoomed_shape = img.shape
            crop_values = (zoomed_shape[0]-original_shape[0],
                            zoomed_shape[1]-original_shape[1],
                            zoomed_shape[2]-original_shape[2])

            img = img[ int(crop_values[0]/2):-int(crop_values[0]/2),
                    int(crop_values[1]/2):-int(crop_values[1]/2),
                    int(crop_values[2]/2):-int(crop_values[2]/2) ]
        return img
    
    def __augment_data(self, case_path, img_file_name):
        # load nibabel Method
        img_path = os.path.join(case_path, img_file_name);
        img_original = nib.load(img_path).get_fdata()

        img_original = self.__crop_img(img_original)
        
        angles = [-10, 10]
        axes_list = [(0,1),(1,2),(0,2)]
        i = 0
        for angle in angles:
            img = np.copy(img_original)
            if self.flip:
                do_flip = random.randint(0, 10)
                if do_flip >=5:
                    axes = random.choice(axes_list)
                    img = np.flip(img, axes)

            axes = random.choice(axes_list)
            img = ndimage.rotate(img, angle, axes=axes, reshape=True)

            if self.zoom > 1.0:
                img = self.make_zoom(img)

            img = self.__crop_img(img)
            
            i += 1
            img = nib.Nifti1Image(img, np.eye(4))
            output_path = os.path.join(case_path, img_file_name[:-7] + str(i) + '.nii.gz')
            nib.save(img, output_path)
            print('Saved! ' + output_path) 

    def data_generation(self):
         with concurrent.futures.ThreadPoolExecutor() as executor:
            # Generate data
            for c, i in enumerate(self.list_IDs): #count, element
                case_path = os.path.join(self.data_path, self.Y_labels[c])

                executor.submit(self.__augment_data, case_path=case_path, img_file_name=i)


project_dir = "/home/pmeslaf/TFM/DATA/FIRST_VISIT_DATA/"

generador = DataAugmentation(data_path=project_dir + '/Train/',
                            classes = ["AD", "CN", "MCI"],
                            flip=True,
                            zoom=1.5)
generador2 = DataAugmentation(data_path=project_dir + '/Validation/',
                            classes = ["AD", "CN", "MCI"],
                            flip=True,
                            zoom=1.5)

# generador.data_generation()
generador2.data_generation()
