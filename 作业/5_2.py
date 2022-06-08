#!/usr/bin/env python
# coding: utf-8

# # 作业第五周（2）CNN网络练习

# 1.仿照课件关于deepdream的程序，在data目录选择一张背景图片(zju1.jpg或zju2.jpg或zju3.jpg或zju4.jpg或者用代码下载一张网络图片保存在save/目录)，
# 选取一个ImageNet预训练网络，通过选择以及组合不同的特征层，训练出一张自己满意的deepdream图片。<BR>
# 
# 

# In[1]:


#首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(    physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


# #### 定义相关函数

# In[2]:


from IPython import display
import PIL.Image
import numpy as np
from tensorflow import keras
from keras.preprocessing import image

# normalize
def normalize_image(img):
    img = 255 * (img+1.0) / 2.0
    return tf.cast(img, tf.uint8)

# visualize
def show_image(img):
    display.display(PIL.Image.fromarray(np.array(img)))
    
# save image
def save_image(img, file_name):
    PIL.Image.fromarray(np.array(img)).save(file_name)


# #### 导入图片

# In[3]:


def read_image(file_name,max_dim=None):
    img = PIL.Image.open(file_name)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# In[4]:


import os
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

img_path = "./data/zju1.jpg"
print(os.path.isfile(img_path))
ori_img = read_image(img_path, 500)
show_image(ori_img)

# noise image
noise_img = np.random.uniform(-1, 1, size=(300, 300, 3)) + 100
# blank image
blank_img = np.zeros(shape=(300, 300, 3)) + 100


# #### 加载预训练模型

# In[5]:


base_model = keras.applications.InceptionV3(include_top=False, weights="imagenet")
base_model.summary()


# In[15]:


layer_names = "conv2d_30"
layers = base_model.get_layer(layer_names).output
layers


# #### 创建特征提取模型

# In[16]:


dream_model = keras.Model(inputs=base_model.input, outputs=layers)
dream_model.summary()


# #### 定义损失函数

# In[17]:


def calc_loss(img, model):
    channel = 13
    img = tf.expand_dims(img, axis=0)
    layer_activations = model(img)
    act = layer_activations[:, :, :, channel]
    loss = tf.math.reduce_mean(act)
    return loss


# #### 图像进化过程

# In[18]:


def render_deepdream(model, img, steps=100, step_size=0.01, verbose=1):
    for n in tf.range(steps):
        with tf.GradientTape() as tape:
        #对1mg进行佛度变换
            tape.watch(img)
            loss = calc_loss(img, model)
        
        #计算损失相对于榆入图像像素的梯度
        gradients = tape.gradient(loss,img)

        #归一化梯度值
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        #在梯度上升中，损失值越来越大，因此可以直接添加损失值到图像中，因为它们的shap相同
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

        #输出过程提示信息
        if (verbose ==1):
            if((n+1) % 10 == 0):
                print ("Step {}/{}, loss {}".format(n+1, steps, loss))
    return img


# In[19]:


import time

img = keras.applications.inception_v3.preprocess_input(ori_img)
img = tf.convert_to_tensor(img) 

start = time.time()
dream_img = render_deepdream(dream_model, img, steps=200, step_size=0.01)
end = time.time()
print(f"total time: {end-start}")

dream_img = normalize_image(dream_img)
show_image(dream_img)


# #### 多通道

# In[20]:


def calc_loss(img, model):
    channels = [13, 109]
    img = tf.expand_dims(img, axis=0)
    layer_activations = model(img)
    
    losses = []
    for ch in channels:
        act = layer_activations[:, :, :, ch]
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)


# In[21]:


import time

img = keras.applications.inception_v3.preprocess_input(ori_img)
img = tf.convert_to_tensor(img) 

start = time.time()
dream_img = render_deepdream(dream_model, img, steps=200, step_size=0.01)
end = time.time()
print(f"total time: {end-start}")

dream_img = normalize_image(dream_img)
show_image(dream_img)


# #### 多层且全部通道综合

# In[22]:


layer_names = ["mixed3", "mixed5"]
layers = [base_model.get_layer(name).output for name in layer_names]
dream_model = keras.Model(inputs=base_model.input, outputs=layers)


# In[23]:


def calc_loss(img, model):
    img = tf.expand_dims(img, axis=0)
    layer_activations = model(img)
    
    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)
    


# In[24]:


import time

img = keras.applications.inception_v3.preprocess_input(ori_img)
img = tf.convert_to_tensor(img) 

start = time.time()
dream_img = render_deepdream(dream_model, img, steps=200, step_size=0.01)
end = time.time()
print(f"total time: {end-start}")

dream_img = normalize_image(dream_img)
show_image(dream_img)


# #### 效果优化

# In[25]:


import time
start = time.time()
OCTAVE_SCALE = 1.30

img = keras.applications.inception_v3.preprocess_input(ori_img)
img = tf.convert_to_tensor(img) 

base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)

for n in range(-2, 3):
    new_shape = tf.cast (float_base_shape * (OCTAVE_SCALE**n), tf.int32)
    img = tf.image.resize(img, new_shape)
    img= render_deepdream(dream_model, img, steps=50, step_size=0.005)
    
end = time.time()
print(f"total time:{end-start}")

img = tf.image.resize(img, base_shape)
dream_img = normalize_image(img)
show_image(dream_img)


# In[ ]:




