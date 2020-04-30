import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
#RUN THIS SECOND
#loads data
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
#creates CNN

model = Sequential()
#Add inputs
model.add(Conv2D(128, (10,10), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(5,5)))
#Add Hidden Layers
model.add(Conv2D(64, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (4,4)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Final layer
model.add(Conv2D(32, (2,2)))
model.add(Activation("relu"))

#outputs
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(3))

#compile model
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#fit model
model.fit(X, y, batch_size=16,epochs=50,validation_split=.1)