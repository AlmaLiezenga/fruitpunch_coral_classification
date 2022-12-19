# -*- coding: utf-8 -*-
"""
@author: Eric

Trains a simple convnet
"""
#%% Load modules
import keras
from keras.models import Sequential
from keras.layers import Rescaling
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.vis_utils import plot_model
 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt

# visualize model
import visualkeras
from PIL import ImageFont

print(K._get_available_gpus())

#%% inputs
batch_size = 32
num_classes = 5
epochs = 20

# path: save model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'coral_reef_cnn.h5'

# path for inputs
path = os.getcwd() + '/data/clean/split/on-func/'
train_folder = path + 'train'
test_folder = path + 'test'
val_folder = path + 'valid'

#%% Load data, split between train, test and valid sets

# normalization layer -> rescaling rgb img in range [0,1]
normalization_layer = Rescaling(1./255)

# train dataset
train_ds = keras.utils.image_dataset_from_directory(
    train_folder,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    image_size=(100, 100),
    batch_size = batch_size,
    shuffle=True
)
train_ds_norm = train_ds.map(lambda x, y: (normalization_layer(x), y))

# test dataset
test_ds = keras.utils.image_dataset_from_directory(
    test_folder,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    image_size=(100, 100),
    batch_size = batch_size,
    shuffle=True
)
test_ds_norm = test_ds.map(lambda x, y: (normalization_layer(x), y))

# val dataset
val_ds = keras.utils.image_dataset_from_directory(
    val_folder,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    image_size=(100, 100),
    batch_size = batch_size,
    shuffle=True
)
val_ds_norm = val_ds.map(lambda x, y: (normalization_layer(x), y))

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
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))       
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.35))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

#%% build model
model = cnn_model()
model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# visualize the model
font = ImageFont.truetype("arial.ttf", 12)
visualkeras.layered_view(model, legend=True, font=font, spacing=10)

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
history = model.fit(train_ds_norm,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_ds_norm),
                    shuffle=True)

#%% save model
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#print(history.history)

#%% evaluate model
score = model.evaluate(val_ds_norm, verbose=1)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Plots
fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot(*[2,1,1])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
ax.set_title('accuracy of CNN')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

ax = fig.add_subplot(*[2,1,2])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss of CNN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
plt.show()