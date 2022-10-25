import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np


# classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #    'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('48','49','50','51','52','53','54','55','56','57',
          '65','66','67','68','69','70',
          '71','72','73','74','75','76','77','78','79','80',
          '81','82','83','84','85','86','87','88','89',
          '90','97','98',
          '100','101','102','103',
          '104','110','113','114','116')
def plot_confusion(y_true, y_pred, val):
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
    #                  columns = [i for i in classes])
    
    plt.figure(figsize = (60,35))
    sn.heatmap(cf_matrix, annot=True)
    plt.savefig('plots/confusion_matrix/'+val+'.png')
    plt.clf()


def plot_accuracies(history, val):
    accuracies = [x['val_acc'] for x in history]
    plt.figure(figsize = (12,7))
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

    plt.savefig('plots/accuracy_epochs/'+val+'.png')
    plt.clf()

def plot_losses(history, val):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('plots/loss/'+val+'.png')
    plt.clf()


def plot_lrs(history, val):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.savefig('plots/learning_rate/'+val+'.png')
    plt.clf()


