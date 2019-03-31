# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
import cv2
import os
from keras.preprocessing import image
from keras.models import load_model
model = load_model('pneumonia-model&weights.h5')
model.summary()

model.load_weights('pneumonia-weights.h5')

model.get_weights()
test_image = image.load_img('30952_img.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

#Class Indices
training_set.class_indices


if result[0][0] == 1:
    prediction = 'Pothole'
else:
    prediction = 'Road'


print(prediction)