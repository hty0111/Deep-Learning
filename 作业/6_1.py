#!/usr/bin/env python
# coding: utf-8

# # 作业第6周（1）循环神经网络练习

# 1.仿照课件关于IMDb数据集的分类训练，在课件示例模型基础上
# 改用GRU、优化网络层数与其它参数，提升分类准确率。<BR>

# In[18]:


#首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# #### 加载数据

# In[19]:


from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np

max_features = 20000
maxlen = 380
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train),"Training sequences")
print(len(x_val),"Validation sequences")


# In[20]:


len(x_train[0]), len(x_train[1]), y_train[0], y_train[1]


# In[21]:


word_index = keras.datasets.imdb.get_word_index (path='imdb_word_index.json')
word_index


# In[22]:


index_to_word = {v:k for k,v in word_index.items()}
index_to_word


# In[23]:


" ".join([index_to_word[x] for x in x_train[0]])


# #### 文字等长处理

# In[24]:


x_train_pad = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val_pad = sequence.pad_sequences(x_val, maxlen=maxlen)


# #### 建立模型

# In[30]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


# In[31]:


def gru():
    model = Sequential()
    model.add(Embedding(output_dim=300, input_dim=max_features, input_length=maxlen))
    model.add(Bidirectional(GRU(units=128,return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(GRU(units=64)))
    model.add(Dropout(0.4))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4, decay=1e-3), metrics=["accuracy"])
    return model


# In[32]:


model = gru()
checkpointer = ModelCheckpoint(filepath="./save/3190105708_6_1_gru_latest.h5", monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
history = model.fit(x_train_pad, y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.2, callbacks=[checkpointer])


# In[33]:


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


# In[34]:


model = gru()
model.load_weights("save/3190105708_6_1_gru_latest.h5")
model.evaluate(x_val_pad, y_val, verbose=1)


# In[ ]:




