import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# load dataset
(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.mnist.load_data(path="mnist.npz")

# scale dataset
train_images, test_images = train_images/255.0, test_images/255.0

# validation split
val_size = len(test_images)
val_idx = np.random.randint(0, len(train_labels), size=len(train_labels))
val_images, val_labels = train_images[val_idx[:val_size]
                                      ], train_labels[val_idx[:val_size]]
train_images, train_labels = train_images[val_idx[val_size:]
                                          ], train_labels[val_idx[val_size:]]

# define model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(x=train_images, y=train_labels, epochs=5)

# validate model
val_error, val_acc = model.evaluate(val_images, val_labels)
print(f"Validation accuracy = {val_acc*100:.2f}%")

# evaluate model
preds = np.argmax(model.predict(test_images), axis=1)
print(f"Accuracy = {np.sum(preds==test_labels)/len(test_labels)*100}%")
