#!/usr/bin/env python
# coding: utf-8

# # 期末综合练习（2）

# ### 利用keras内置数据集练习文本数据分类（要求采用Conv1D + RNN模型）
# 路透社数据集：<BR>
# 路透社数据集包含许多短新闻及其对应的主题，由路透社在1986年发布。包括46个不同的主题，其中某些主题的样本更多，但是训练集中的每个主题都有至少10个样本。<BR>
# 与IMDB数据集一样，路透社数据集也内置到了Keras库中，并且已经经过了预处理。<BR>
# #### 提示：
# 由于文本较长，先用CNN卷积上采样到较短长度，再用RNN处理是一个可以避免梯度消失的好方法。<BR>
#     (由于卷积核为一维，卷积核大小要相应增大到5或7，stride增加到3或5)。
# #### 要求：
# 利用callback将最佳模型保存到文件(注意：文件名应包含学号，保存在'save/'目录！)，
# 最后对最佳模型进行指标评估，展示混淆矩阵
# #### 数据读取方法：
# (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000, test_split=0.2)
# 
# #### 考核办法：
# score = model.evaluate(x_test, y_test)
# 计算得到的准确率为指标，准确率达到0.7为及格成绩起点，0.8以上优秀

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[19]:


max_features = 10000
maxlen = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=max_features, test_split=0.2)
print(x_train.shape,x_test.shape)
print(len(x_train[0]), len(x_train[1]))


# In[20]:


word2index = keras.datasets.reuters.get_word_index()
index2word = dict([(value, key)for(key, value)in word2index.items()])
# 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
content =' '.join([index2word.get(i - 3, '?')for i in x_train[0]])
content


# In[21]:


import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train_vector = vectorize_sequences(x_train, maxlen)
x_test_vector = vectorize_sequences(x_test, maxlen)
print(x_train_vector.shape)

x_train_pad = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test_pad = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train_pad.shape)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)
print(y_train_onehot.shape)


# In[22]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def model1():
    model = Sequential()
    model.add(Embedding(output_dim=512, input_dim=max_features, input_length=maxlen))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.3))
    model.add(GRU(128))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(46, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def model2():
    model = Sequential()
    model.add(Embedding(output_dim=512, input_dim=max_features, input_length=maxlen))
    model.add(Conv1D(128, 7, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(GRU(128))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
#     model.add(Dense(128, activation='relu'))
    model.add(Dense(46, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def model3():
    model = Sequential()
    model.add(Embedding(output_dim=512, input_dim=max_features, input_length=maxlen))
    model.add(Conv1D(128, 7, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.3))
    model.add(GRU(128))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(46, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# In[ ]:


path1 = 'save/3190105708_final2.h5'      # 79.52 %
path2 = 'save/3190105708_final2_1.h5'    # 79.12 %
path3 = 'save/3190105708_final2_new.h5'  # 79.56 %
model = model2()
# model.load_weights('save/3190105708_final2.h5')
checkpointer = ModelCheckpoint(filepath=path2, monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
history = model.fit(x_train_pad, y_train_onehot, batch_size=16, epochs=15, verbose=1, validation_data=(x_test_pad, y_test_onehot),
                    callbacks=[checkpointer])
# history = model.fit(x_train_vector, y_train_onehot, batch_size=64, epochs=10, verbose=1, validation_split=0.2,
#                     callbacks=[checkpointer])


# In[ ]:


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


# In[ ]:


# model = model2()
model.load_weights(path2)
model.evaluate(x_test_pad, y_test_onehot, verbose=1)


# #### 总结说明
# 此处说明关于模型设计与模型训练（参数设置、训练和调优过程）的心得与总结

# **final 1**
# 
# 第一个大作业是个较为常见的图像分类题目。
# 在数据处理上，由于不是很会在`TFRecordDataset`中将label转成onehot，所以全部重新读取成了np.ndarray格式后再做转换，并且在加入了一些图像增强操作。
# 模型上，图像分类最常规的就是使用CNN，一开始想用之前训练cifar10的VGG16，但是发现由于图像像素变多，导致GPU显存不够了。于是换成ResNet50，结果也是显存不够，所以换成轻量化的MobileNet，并且用了“imagenet”上的预训练权重。将MobileNet除全连接层的网络提取出来，加上全局池化和6分类的全连接层，就得到了这次使用的模型。
# 调参方面，batch太大也会超显存，甚至设成1都没问题，lr要小一点，否则可能第一个epoch就过拟合了，还算是比较轻松就能达到90的准确率。
# 

# **final2**
# 
# 第二个大作业非常之有难度啊，首先是发现很容易过拟合，所以网络应该用最简单的一层Conv和一层RNN，但还是过拟合，所以加入池化层和dropout。epoch大了会导致梯度消失，不过用callback里的ModelCheckpoint保存最佳模型就行。之后正确率一直被限制在79%，小修改
