import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
#RUN THIS FIRST

#size of images
WIDTH_SIZE = 300
HEIGHT_SIZE = 250

#creates categories for labels
CATEGORIES = []
for CATEGORY in os.listdir('./TRAINING'):
    CATEGORIES.append(CATEGORY)
DATADIR = "./TRAINING"
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #makes img grayscale
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                #resize image
                new_array = cv2.resize(img_array,(WIDTH_SIZE,HEIGHT_SIZE))
                #add image data to training list
                training_data.append([new_array,class_num])
            except:
                pass

create_training_data()

random.shuffle(training_data)
#creates data for CNN
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, WIDTH_SIZE, HEIGHT_SIZE, 1)
y = np.array(y)


#saves data as pickle file
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)