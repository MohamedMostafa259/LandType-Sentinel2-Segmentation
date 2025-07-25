import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
import itertools
import math


##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### 


class PlotDiagnostics(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []

    self.losses = []
    self.val_losses = []

    self.jaccard_coef = []
    self.val_jaccard_coef = []

    self.fig = plt.figure()
    self.logs = []

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)

    self.losses.append(logs.get('loss'))
    self.val_losses.append(logs.get('val_loss'))

    self.jaccard_coef.append(logs.get('jaccard_coef'))
    self.val_jaccard_coef.append(logs.get('val_jaccard_coef'))

    self.i += 1

    plt.figure(figsize=(14,8))
    f, (graph1, graph2) = plt.subplots(1,2, sharex=True)
    
    clear_output(wait=True)

    graph1.set_yscale('log')
    graph1.plot(self.x, self.losses, label="loss")
    graph1.plot(self.x, self.val_losses, label="val_loss")
    graph1.legend()

    graph2.set_yscale('log')
    graph2.plot(self.x, self.jaccard_coef, label="jaccard_coef")
    graph2.plot(self.x, self.val_jaccard_coef, label="val_jaccard_coef")
    graph2.legend()

    plt.show()


##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### 


def show_final_history(history): 
    """Automatically plots all training/validation metrics from a Keras history object, two plots per row.""" 
 
    history_dict = history.history 
    metrics = sorted(set(k.replace('val_', '') for k in history_dict if not k.startswith('val_'))) 
    num_metrics = len(metrics)
    
    # Compute number of rows needed if 2 plots per row
    num_cols = 2
    num_rows = math.ceil(num_metrics / num_cols)
    
    plt.style.use("ggplot") 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case it's 2D
    
    for idx, metric in enumerate(metrics): 
        ax = axes[idx]
        ax.plot(history_dict[metric], 'r-', label=f'Training {metric}') 
        val_key = f'val_{metric}' 
        if val_key in history_dict: 
            ax.plot(history_dict[val_key], 'g-', label=f'Validation {metric}') 
 
        ax.set_title(metric.capitalize()) 
        ax.set_xlabel('Epochs') 
        ax.set_ylabel(metric) 
        ax.legend(loc='best')

    # Hide unused subplots if num_metrics is odd
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### 


# src: https://www.kaggle.com/code/apollo2506/eurosat-allbands-classification
def plot_learning_rate(history):
    plt.style.use("ggplot")
    plt.plot(np.arange(0, len(history.history['learning_rate'])), history.history['learning_rate'])
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.show()


##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### 


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2.0
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        plt.text(j,i, format(cm[i,j], fmt),
                horizontalalignment = 'center',
                color = "white" if cm[i,j] > thresh else "black")
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.grid(False)


##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### ----- ##### 


