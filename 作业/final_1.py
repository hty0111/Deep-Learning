#!/usr/bin/env python
# coding: utf-8

# # 期末综合练习（1）

# ### 改造CIFAR分类练习，对某图像数据集进行分类（模型形式不限）
# Context：<BR>
# This is image data of Natural Scenes around the world.<BR>
# Content：<BR>
# This Data contains around 25k images of size 96x96 distributed under 6 categories.<BR>
# {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2,
# 'mountain' -> 3,  'sea' -> 4, 'street' -> 5 }
# <BR>
# 为方便使用，已将数据集转为tfrecords格式文件<BR>
#     ( 可以下载到本地进行调试练习: http://10.13.23.218/chen/deepl/  )。<BR>
#     
# #### 要求：
# 利用callback将最佳模型保存到文件(注意：文件名应包含学号，保存在'save/'目录！)，
# 最后对最佳模型进行指标评估，展示混淆矩阵
# 
# #### 考核办法：
# score = model.evaluate(testset)
# 计算得到的准确率为指标，达到0.8为及格成绩起点，0.9优秀
#### 数据读取方法：
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size=64
trainset=(tf.data.TFRecordDataset("data/intel-trainset.tfrecords")
        .map(parse_tfrecord_fn)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
         )
         
testset=(tf.data.TFRecordDataset("data/intel-testset.tfrecords")
        .map(parse_tfrecord_fn)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
         )
    
model.fit(trainset, validation_data=testset, ...)  #不区分验证集与测试集<BR>
关于数据增强，可以选择嵌入模型，也可以选择CPU预处理
CPU预处理API参考：https://www.tensorflow.org/api_docs/python/tf/image
示例：
    image = tf.image.random_brightness(image, max_delta=0.5) # 随机增加亮度50%
    image = tf.image.random_flip_left_right( image)
    image = tf.image.random_flip_up_down( image)

# In[1]:


#首先执行GPU资源分配代码，勿删除。如果产生CUDA运行错误，一般为GPU使用冲突，过一段时间重试
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])


# In[2]:


def parse_tfrecord_fn(example, aug=False):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
#     print(image.shape)
    example["image"] = tf.image.resize(image, [96, 96])/255.0
#     example["label"] = tf.sparse.to_dense(example["label"])
#     example["label"] = to_categorical(example["label"])
#     print(label.shape)
    if aug:  ##根据需要可以增加预处理，比如添加数据增强
        image = tf.image.random_brightness(image, max_delta=0.5) # 随机增加亮度50%
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    return example["image"], example["label"]

def parse_tfrecord_train(example):
    return parse_tfrecord_fn(example, True)

def parse_tfrecord_test(example):
    return parse_tfrecord_fn(example, False)


# In[3]:


from keras.utils import to_categorical
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size=1
trainset=(tf.data.TFRecordDataset("data/intel-trainset.tfrecords")
        .map(parse_tfrecord_train)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
         )
testset=(tf.data.TFRecordDataset("data/intel-testset.tfrecords")
        .map(parse_tfrecord_test)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
         )

print(trainset)

x_train, y_train = [], []
for x, y in trainset:
    x = np.array(x).reshape(96, 96, 3)
    y = np.array(y).reshape(-1)
    x_train.append(x)
    y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape, y_train.shape)

x_test, y_test = [], []
for x, y in testset:
    x = np.array(x).reshape(96, 96, 3)
    y = np.array(y).reshape(-1)
    x_test.append(x)
    y_test.append(y)
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape, y_test.shape)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)
print(y_train_onehot[0])


# In[4]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def get_label(class_code):
    labels = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}
    return labels[class_code]

fig = plt.figure(figsize=(15, 15))
samples = np.random.randint(0, len(x_train)+1, size=9)
for count, n in enumerate(samples, start=1):
    ax = fig.add_subplot(3, 3, count)
    ax.imshow((x_train[n]*255).astype(np.uint8), interpolation='nearest')
    label_name = "Label:" + str(get_label(int(y_train[n])))
    ax.set_title(label_name, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

# fig.tight_layout()
plt.show()


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, GlobalAveragePooling2D
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications import VGG16, ResNet50, MobileNet, InceptionV3
from tensorflow.keras.utils import plot_model
import pandas as pd

shape = (96, 96, 3)

def resnet50():
    return ResNet50(input_shape=shape, include_top=False, weights="imagenet")

def vgg16():
    return VGG16(input_shape=shape, include_top=False, weights="imagenet")

def mobilenet():
    return MobileNet(input_shape=shape, include_top=False, weights="imagenet")

def inception():
    return InceptionV3(input_shape=shape, include_top=False, weights="imagenet")

def model1(): 
    base_model = mobilenet()
    base_model.trainable = True

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(6, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
    return model

model1().summary()


# In[7]:


path1 = f"./save/3190105708_final1_1.h5"  # 0.9197
path2 = f"./save/3190105708_final1_2.h5"  # 0.9140
model = model1()
checkpointer = ModelCheckpoint(filepath=path2, monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
history = model.fit(x_train,
                    y_train_onehot,
                    batch_size=16,
                    epochs=10,
                    use_multiprocessing=True,
                    validation_data=(x_test, y_test_onehot),
                    callbacks=[checkpointer],
                    verbose=1
                   )


# In[8]:


import matplotlib.pyplot as plt
def show_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

show_history(history, "accuracy", "val_accuracy")
show_history(history, "loss", "val_loss")


# In[11]:


model.load_weights(path1)
model.evaluate(x_test, y_test_onehot, verbose=1)


# In[12]:


prediction = model.predict_classes(x_test)

label_dict = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}
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

plot_images_labels_prediction(x_test, y_test, prediction, idx=0)  


# In[15]:


Predicted_Probability = model.predict(x_test)

def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability,i):
    print('label:',label_dict[y[i][0]], '\npredict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(x_img[i])
    plt.show()
    for j in range(6):
        print(label_dict[j] + 'Probability:%1.9f'%(Predicted_Probability[i][j]))
        
show_Predicted_Probability(y_test, prediction, x_test, Predicted_Probability, 0)


# In[17]:


import pandas as pd
print(label_dict)
pd.crosstab(y_test.reshape(-1), prediction, rownames=["label"], colnames=["predict"])


# In[ ]:




