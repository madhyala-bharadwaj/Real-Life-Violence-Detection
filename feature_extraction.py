import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
import cv2

pretrained_model = InceptionV3()
pretrained_model = Model(inputs=pretrained_model.input,outputs=pretrained_model.layers[-2].output)
def feature_extractor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (299, 299))/255.0
    img = np.expand_dims(frame, axis=0)
    feature_vector = pretrained_model.predict(img, verbose=0)
    return feature_vector