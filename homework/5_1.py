#!/usr/bin/env python
# coding: utf-8

# # 作业第五周（1）CNN网络练习

# 1.在第四周cifar10分类练习基础上，加入数据增强措施，使用callback方法保存最佳模型；训练完成后将保存在文件的模型读入，用于评估与预测测试集样本。

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ### 读取数据集

# In[2]:


from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
np.random.seed (10)
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = cifar10.load_data()
x_train_raw.shape, y_train_raw.shape, x_test_raw.shape, y_test_raw.shape
x_train = x_train_raw.astype("float32") / 255
x_test = x_test_raw.astype("float32") / 255
y_train = np_utils.to_categorical(y_train_raw, 10)
y_test = np_utils.to_categorical(y_test_raw, 10)
y_train[0]


# ### 图像增强

# In[3]:


# image augmentation
from keras.preprocessing.image import ImageDataGenerator

generated_images = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.2, 
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

generated_images.fit(x_train)


# In[4]:


# show images
x_train_subset = np.squeeze(x_train[:10])
 
fig = plt.figure(figsize=(20, 2))
# original images
for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1)
    ax.imshow(x_train_subset[i])
fig.suptitle('Original Images', fontsize=20)
plt.show()
 
# augmented images
fig = plt.figure(figsize=(20, 2))
for x_batch in generated_images.flow(x_train_subset, batch_size=12, shuffle=False):
    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        ax.imshow(x_batch[i])
    fig.suptitle('Augmented Images', fontsize=20)# 总标题
    plt.show()
    break


# ### 训练
# 第四周（2）作业中已经用了图像增强并且保存了模型，这里加入ModelCheckpoint用callback保存最佳模型

# In[10]:


# VGG
from keras.callbacks import ModelCheckpoint

lr = 0.1
lr_drop = 20
batch_size = 128
epochs = 250
weight_decay = 0.0005

def lr_scheduler(epoch):
    return lr * (0.5 ** (epoch // lr_drop))
reduce_lr = LearningRateScheduler(lr_scheduler)

def VGG():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3),kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

model = VGG()
save_path = f"./save/3190105708_5_1_vgg_{epochs}epochs.h5"
checkpointer = ModelCheckpoint(filepath=save_path, monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[12]:


history = model.fit_generator(generated_images.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  validation_data=(x_test, y_test),
                                  callbacks=[checkpointer, reduce_lr],
                                  verbose=1)
# model.save_weights('save/3190105708_vgg_callback.h5')


# In[15]:


import os
# latest model
print(os.path.isfile(os.path.join(os.getcwd(), save_path)))
# last model
print(os.path.isfile(os.path.join(os.getcwd(), "save/3190105708_vgg.h5")))


# ### 结果对比
# 在上次模型的基础上加入了ModelCheckpoint，预测结果几乎一致，因为已经对学习率进行了调整策略，所以不太可能过拟合

# In[20]:


model.load_weights("save/3190105708_vgg.h5")
scores = model.evaluate(x_test, y_test, verbose=1)
print(f"4_2_model || loss: {scores[0]:.4f}, acc: {scores[1]:.4f}")
model.load_weights(save_path)
scores = model.evaluate(x_test, y_test, verbose=1)
print(f"5_1_model || loss: {scores[0]:.4f}, acc: {scores[1]:.4f}")


# ### 预测错误的结果用红色标出

# In[21]:


prediction = model.predict_classes(x_test)

label_dict={0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(18,18)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx])
        title = str(i+1) + '  ' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += ' --> ' + label_dict[prediction[i]]
            if labels[i][0] != prediction[i]:
                color = "red"
            else: 
                color = "black"
            ax.set_title(title, fontsize=10, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
    plt.show()

plot_images_labels_prediction(x_test_raw, y_test_raw, prediction, idx=0)  


# ### 查看预测概率

# In[22]:


Predicted_Probability = model.predict(x_test)

def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability,i):
    print('label:',label_dict[y[i][0]], 'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_img[i],(32,32,3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + 'Probability:%1.9f'%(Predicted_Probability[i][j]))
        
show_Predicted_Probability(y_test_raw, prediction, x_test_raw, Predicted_Probability, 0)


# ### 混淆矩阵

# In[23]:


import pandas as pd
print(label_dict)
pd.crosstab(y_test_raw.reshape(-1), prediction, rownames=["label"], colnames=["predict"])

