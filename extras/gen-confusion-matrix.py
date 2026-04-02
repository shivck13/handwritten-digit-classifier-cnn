import os

#suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data (Standard MNIST Load)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

# 2. Get Predictions from your Model
# Assuming 'model' is your trained CNN
model = tf.keras.models.load_model('model.h5')  # Load your trained model
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 3. Compute Confusion Matrix using TensorFlow
# This replaces sklearn.metrics.confusion_matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_classes, num_classes=10).numpy()

# 4. Visualization using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix for MNIST Digit Recognition')
plt.colorbar()

# Set ticks for each digit 0-9
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 5. Add Count Labels to Each Cell
# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()