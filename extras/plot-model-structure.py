from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import os

model = models.Sequential([
    layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1), name="CL-1"),
    layers.Conv2D(64, (5,5), activation='relu', name="CL-2"),
    layers.MaxPooling2D((2,2), name="MP-1"),
    layers.Dropout(0.25, name="DO-1"),
    layers.Conv2D(128, (5,5), activation='relu', name="CL-3"),
    layers.MaxPooling2D((2,2), name="MP-2"),
    layers.Dropout(0.25, name="DO-2"),
    layers.Flatten(name="FL-1"),
    layers.Dense(128, activation='relu', name="DL-1"),
    layers.Dropout(0.3, name="DO-3"),
    layers.Dense(64, activation='relu', name="DL-2"),
    layers.Dropout(0.3, name="DO-4"),
    layers.Dense(10, activation='softmax', name="DL-3")
])

model.summary()

file = os.path.join(os.getcwd(), "figs/cnn_model_diagram.png")
plot_model(model, to_file=file, show_shapes=True, show_layer_names=True, dpi=100)
