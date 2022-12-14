import matplotlib.pyplot as plt

def create_plots(history):
    fig = plt.figure(figsize=(5,10))
    ax = fig.add_subplot(*[2,1,1])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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