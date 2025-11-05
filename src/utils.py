

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
  plt.figure(figsize=(12,5))

  plt.subplot(1,2,1)
  plt.plot(history.history['accuracy'])
  if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
  else:
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'], loc='upper left')

  plt.subplot(1,2,2)
  plt.plot(history.history['loss'])
  if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
  else:
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train'], loc='upper left')

  plt.tight_layout()
  plt.show()


def calculate_iou(y_true, y_pred):
  intersection = np.logical_and(y_true, y_pred).sum()
  union = np.logical_or(y_true, y_pred).sum()
  if union == 0:
    return 0.0
  return intersection / union


def calculate_dice(y_true, y_pred):
  intersection = np.logical_and(y_true, y_pred).sum()
  total = np.sum(y_true) + np.sum(y_pred)
  if total == 0:
    return 0.0
  return 2 * intersection / total
