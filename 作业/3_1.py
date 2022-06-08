#!/usr/bin/env python
# coding: utf-8

# # 作业第三周（1）MLP模型练习

# #### 1.仿照课件关于mnist数据集的分类训练，设计一个简单多层感知机网络，训练fashion_mnist的分类操作。
#     (打印loss变化曲线曲线，显示测试集最后的预测准确率、混淆矩阵)

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[2]:


# train & test
from keras.datasets import fashion_mnist
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = fashion_mnist.load_data()
x_train_raw.shape


# In[3]:


# normalize and one-hot
from keras.utils import np_utils
x_train = x_train_raw.reshape(60000, 784).astype("float32") / 255
x_test = x_test_raw.reshape(10000, 784).astype("float32") / 255
y_train = np_utils.to_categorical(y_train_raw, 10)
y_test = np_utils.to_categorical(y_test_raw, 10)
y_test[0]


# In[4]:


# model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer="normal"))
model.add(Dense(512, activation='relu', kernel_initializer="normal"))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
model.summary()


# In[5]:


# train
history = model.fit(x_train, y_train, batch_size=128, epochs=30, verbose=0, validation_split=0.2)


# In[6]:


# visualize
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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


# In[7]:


# test
scores = model.evaluate(x_test, y_test)
scores


# In[8]:


# predict
prediction = model.predict_classes(x_test)
prediction


# In[9]:


# convert label_id to label_text
def get_fashion_mnist_labels(label_id):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels[label_id]


# In[10]:


# if mismatch, labels will be shown in red
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
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

plot_images_labels_prediction(x_test_raw, y_test_raw, prediction, idx=110)


# In[11]:


import pandas as pd
pd.crosstab(y_test_raw, prediction, rownames=['label'], colnames=['predict'])

