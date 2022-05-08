import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import os
import random
from sklearn.metrics import confusion_matrix, classification_report
import cv2

DATADIR = r".\DL_DATA"
CATEGORIES = ["Book", "Cardboard_box", "Cup"]

def make_train_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                train_data.append([img_array, class_num])
            except Exception as e: 
                pass
train_data = []
make_train_data()
random.shuffle(train_data)
print(len(train_data))

X = []
y = []

for features, label in train_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 100, 100, 1)
X = X/255.0
y = np.array(y)

class tooGood(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("accuracy") > 0.9):
            print("\n Desired accuracy reached; training ended.")
            self.model.stop_training = True
callbacks = tooGood()

cnn_model = models.Sequential([
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", input_shape = X.shape[1:]),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(640, activation = "relu"),
    layers.Dense(3, activation = "softmax")
    
    #Maybe add another Dense 1 and activation? Maybe activation sigmoid? 
    
])

cnn_model.compile(optimizer = "adam", 
             loss = "sparse_categorical_crossentropy", 
            #Maybe do binary_crossentropy? if it's greyscale... 
             metrics = ["accuracy"])
#Remember to change the val split
cnn_model.fit(X, y, batch_size = 20, validation_split = 0.1, epochs = 10, callbacks = [callbacks])

cnn_model.save(r".\model_data")
