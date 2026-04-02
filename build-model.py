from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# file name to save the trained model
MODEL_NAME = "model.h5"

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and expand dimensions for CNN
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation to simulate real-world handwriting conditions
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Build the CNN model with convolutional, pooling, dropout, and dense layers
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

# Compile the model with optimizer, loss function, and metrics
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with augmented data
batch_size = 64
epochs = 10
model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_test, y_test)
)

# Save the trained model
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")