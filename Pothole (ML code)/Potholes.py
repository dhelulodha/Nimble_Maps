import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import cv2

#Processsed Dataset path
PP_PATH = os.path.join(os.getcwd(), '../PotholeImageDataset_pp/')

# =============================================================================
# Data Pre-processing (Dont run if using PotholesImageDataset_pp)
# =============================================================================
#Classes
PATH_1 =  'AsphaltPavement/'
PATH_2 =  'Manhole/'
PATH_3 =  'Pothole/'
PATH_4 =  'RoadMarking/'
PATH_5 =  'Shadow/'



#Resizing and loading    
def read_img(img_path):
    img = cv2.imread(img_path,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    return img

for img_path in tqdm(os.listdir(PATH_1)):
    cv2.imwrite(os.path.join(PP_PATH,PATH_1,img_path),read_img(PATH_1 + img_path))

for img_path in tqdm(os.listdir(PATH_2)):
    cv2.imwrite(os.path.join(PP_PATH,PATH_2,img_path),read_img(PATH_2 + img_path))

for img_path in tqdm(os.listdir(PATH_3)):
    cv2.imwrite(os.path.join(PP_PATH,PATH_3,img_path),read_img(PATH_3 + img_path))

for img_path in tqdm(os.listdir(PATH_4)):
    cv2.imwrite(os.path.join(PP_PATH,PATH_4,img_path),read_img(PATH_4 + img_path))

for img_path in tqdm(os.listdir(PATH_5)):
    cv2.imwrite(os.path.join(PP_PATH,PATH_5,img_path),read_img(PATH_5 + img_path))




# =============================================================================
# Neural Network model
# =============================================================================

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing CNN
classifier = Sequential()

#Convolution layer1
classifier.add(Conv2D(128, kernel_size = (3, 3), activation='relu', input_shape=(128, 128, 3)))
#Pooling Layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Convolution layer2
classifier.add(Conv2D(128, kernel_size = (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Convolution layer3
classifier.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#Hidden Layer
classifier.add(Dense(output_dim=128,activation='relu'))
#Output layer
classifier.add(Dense(output_dim=5,activation='softmax'))

#Compiling CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( PP_PATH+'train_set/',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory(  PP_PATH+'test_set/',
                                               target_size=(128, 128),
                                               batch_size=32,
                                               class_mode='categorical')
#Fitting
classifier.fit_generator(
        training_set,
        steps_per_epoch=7452/32,
        epochs=20,
        validation_data=test_set,
        validation_steps=570/32)

#Saving Model & Weights
classifier.save('Potholes-model.h5')
classifier.save_weights('Potholes-weights.h5')
