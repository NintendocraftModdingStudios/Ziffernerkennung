import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Laden und Aufteilen des MNIST-Datensatzes
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Daten vorverarbeiten und normalisieren
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255.0

# One-Hot-Encoding der Labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Neuronales Netzwerk erstellen
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Modell kompilieren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modell trainieren
model.fit(x_train, y_train, epochs=15, batch_size=128, validation_data=(x_test, y_test))

# Modell speichern
model.save('mnist_model.h5')

