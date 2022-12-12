# -*- coding: utf-8 -*-
"""
@author: Eric

Trains a simple convnet
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import os

from sklearn.metrics import classification_report, confusion_matrix
from functions import create_plots, plot_confusion_matrix

batch_size = 128
num_classes = 6
epochs = 10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'coral_reef_cnn.h5'

#%% the data, split between train and test sets
image = 
label = 
x_train = image[]
x_test = image[]

y_train_vector = label[]
y_test_vector = label[]

#%%
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_vector, num_classes)
y_test = keras.utils.to_categorical(y_test_vector, num_classes)

#%%
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))       
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

#%%
model = cnn_model()
model.summary()
#%%
# optimizer
opt = keras.optimizers.Adam()

# Let's train the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#%%
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# save model
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Plots
class_names = ['Algae', 'Hard Coral', 'Soft Coral', 'Other Invertebrates', 'Other']
def plot_conf_matrix(test):
    if not test: 
        Y_pred = model.predict(x_train,verbose=2)
        y_pred = np.argmax(Y_pred,axis=1)
        for ix in range(6):
            print (ix, confusion_matrix(np.argmax(y_train,axis=1), y_pred)[ix].sum())
        print (confusion_matrix(np.argmax(y_train,axis=1), y_pred))    
        plot_confusion_matrix(confusion_matrix(np.argmax(y_train,axis=1), y_pred), classes=class_names)
    else:
        
        Y_pred = model.predict(x_test,verbose=2)
        y_pred = np.argmax(Y_pred,axis=1)
        for ix in range(6):
            print (ix, confusion_matrix(np.argmax(y_test,axis=1), y_pred)[ix].sum())
        print (confusion_matrix(np.argmax(y_test,axis=1), y_pred))    
        plot_confusion_matrix(confusion_matrix(np.argmax(y_test,axis=1), y_pred), classes=class_names)
#%%

plot_conf_matrix(True)
#%%
create_plots(history)