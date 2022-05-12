# 首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import
from keras.utils import np_utils
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, AvgPool2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, Lambda, BatchNormalization, InputLayer
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = fashion_mnist.load_data()
# reshape for net
x_train = x_train_raw.reshape((x_train_raw.shape[0],) + (28, 28, 1)).astype('float32') / 255
x_test = x_test_raw.reshape((x_test_raw.shape[0],) + (28, 28, 1)).astype('float32') / 255


# model 1 | test scores: 0.926
y_train = np_utils.to_categorical(y_train_raw, 10)
y_test = np_utils.to_categorical(y_test_raw, 10)

model1 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), kernel_initializer='he_normal'),
    MaxPooling2D(pool_size=2),
    Dropout(0.25),
    Conv2D(64, kernel_size=3, activation='relu'),
    Dropout(0.25),
    Conv2D(128, kernel_size=3, activation='relu'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history1 = model1.fit(x_train, y_train, batch_size=512, epochs=50, verbose=1, validation_split=0.2)


# model 2 | test scores: 0.9103
y_train = y_train_raw.reshape(-1)
y_test = y_test_raw.reshape(-1)
early_stop = EarlyStopping(monitor="val_loss", patience=4, verbose=1)

model2 = Sequential()
model2.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding='valid', input_shape=(28, 28, 1)))
model2.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Dropout(0.25)) 
model2.add(Flatten())
model2.add(Dense(128, activation="relu"))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation="softmax"))

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history2 = model2.fit(x_train, y_train, batch_size=512, epochs=50, verbose=1, validation_split=0.2)


# model 3 | test scores: 0.9146
y_train = np_utils.to_categorical(y_train_raw, 10)
y_test = np_utils.to_categorical(y_test_raw, 10)

model3 = Sequential([
    InputLayer(input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(64, (4, 4), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),
    Conv2D(64, (4, 4), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')            
])

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history3 = model3.fit(x_train, y_train, batch_size=512, epochs=50, verbose=1, validation_split=0.2)


# model 4 | test scores: 0.9309
mean_px = x_train.mean()
std_px = x_train.std()
def norm_input(x): 
    return (x-mean_px)/std_px
y_train = y_train_raw.reshape(-1)
y_test = y_test_raw.reshape(-1)

early_stop = EarlyStopping(monitor="val_loss", patience=6, verbose=1)

model4 = Sequential([
    Lambda(norm_input, input_shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28, 1)),
    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),    
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model4.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
history4 = model4.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_split=0.2, callbacks=[early_stop])


# model 5 | test scores: 0.9260
mean_px = x_train.mean()
std_px = x_train.std()
def norm_input(x): 
    return (x-mean_px)/std_px
y_train = y_train_raw.reshape(-1)
y_test = y_test_raw.reshape(-1)

early_stop = EarlyStopping(monitor="val_loss", patience=4, verbose=1)

model5 = Sequential([
    Lambda(norm_input, input_shape=(28,28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),    
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),    
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model5.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
history5 = model5.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_split=0.2)


# visualize
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

# history.history.keys()
plot_history(history4)

# test
# y_test = np_utils.to_categorical(y_test_raw, 10)    # for 'categorical_crossentropy'
y_test = y_test_raw.reshape(-1)    # for 'sparse_categorical_crossentropy'
scores = model4.evaluate(x_test, y_test)
print(f"loss:{scores[0]:.2f}, accuracy:{scores[1]:.2f}")


# predict
prediction = model4.predict_classes(x_test)

# convert label_id to label_text
def get_fashion_mnist_labels(label_id):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels[label_id]

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(18, 18)
    fig.tight_layout()
    if num > 25:
        num = 25
    for i in range(0, num):
        ax=plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx])
        title = "label=" + get_fashion_mnist_labels(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + get_fashion_mnist_labels(prediction[idx])
        if labels[idx] != prediction[idx]:
            color = "red"
        else: 
            color = "black"
        ax.set_title(title, fontsize=10, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

plot_images_labels_prediction(x_test_raw, y_test_raw, prediction, idx=100)


