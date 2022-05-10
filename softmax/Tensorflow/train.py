import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# device
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# normalize and one-hot
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = fashion_mnist.load_data()
x_train = x_train_raw.reshape(60000, 784).astype("float32") / 255
x_test = x_test_raw.reshape(10000, 784).astype("float32") / 255
y_train = np_utils.to_categorical(y_train_raw, 10)
y_test = np_utils.to_categorical(y_test_raw, 10)
y_test[0]

# model
model = Sequential()
model.add(Dense(256, activation='tanh', kernel_initializer = 'he_normal' ,input_shape=(28*28,)))
model.add(Dropout(0.4))
model.add(Dense(128, activation='tanh',kernel_initializer = 'he_normal'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='tanh',kernel_initializer = 'he_normal'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='sigmoid',kernel_initializer = 'he_normal'))
optimizer = SGD(lr=0.01, momentum=0.975, decay=2e-06, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# early_stop = EarlyStopping(monitor="val_loss", patience=4, verbose=1)
history = model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.2)

# visualize
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
plot_history(history)

# test
scores = model.evaluate(x_test, y_test)
print(f"loss:{scores[0]:.4f}, accuracy:{scores[1]:.4f}")

# predict
prediction = model.predict_classes(x_test)

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
