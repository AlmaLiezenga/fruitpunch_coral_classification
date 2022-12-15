# -*- coding: utf-8 -*-
"""
@author: Eric

Trains a simple convnet
"""
#%% Load modules
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from functions import create_plots

print(K._get_available_gpus())

#%% inputs
batch_size = 32
num_classes = 5
epochs = 10

# path: save model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'coral_reef_cnn.h5'

# path for inputs
path = os.getcwd() + '/data/clean/split/on-func/'
train_folder = path + 'train'
test_folder = path + 'test'
val_folder = path + 'valid'

#%% Load data, split between train, test and valid sets
train_ds = keras.utils.image_dataset_from_directory(
    train_folder,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    image_size=(100, 100)
)

test_ds = keras.utils.image_dataset_from_directory(
    test_folder,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    image_size=(100, 100)
)

val_ds = keras.utils.image_dataset_from_directory(
    val_folder,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    image_size=(100, 100)
)

class_names = train_ds.class_names
print(class_names)

#%% model arch
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(100,100,3)))
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
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

#%% build model
model = cnn_model()
model.summary()

#%% optimizer, metrics and compile
opt = keras.optimizers.Adam(learning_rate=0.0001)

# metrics
metrics = ['accuracy',
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall')
           ]

# compile
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=metrics)

#%% Let's train the model 
history = model.fit(train_ds,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(val_ds))

#%% save model
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

print(history.history)

#%% evaluate model
score = model.evaluate(test_ds, verbose=1)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Plots
create_plots(history)