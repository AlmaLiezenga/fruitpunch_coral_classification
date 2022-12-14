import matplotlib.pyplot as plt
import numpy as np 
import itertools
from sklearn.metrics import classification_report, confusion_matrix

def create_plots(history):
    fig = plt.figure(figsize=(5,10))
    ax = fig.add_subplot(*[2,1,1])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    ax.set_title('accuracy of CNN')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
   
    
    ax = fig.add_subplot(*[2,1,2])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss of CNN')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    #plt.clf()
    plt.show()