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
    
def plot_confusion_matrix(confusionmatrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        confusionmatrix = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusionmatrix)

    plt.imshow(confusionmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, format(confusionmatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusionmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')