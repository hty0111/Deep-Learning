#!/usr/bin/env python
# coding: utf-8

# # 作业第7周：GAN练习

# 1.仿照课件示例的GAN生成网络，复现Fashion_mnist数据集的GAN生成效果或Mnist数据集的GAN生成效果。<BR>（学有余力同学可以挑战一下anime faces的GAN生成，建议使用WGAN）
# 

# In[19]:


#首先执行GPU资源分配代码，勿删除。
import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])


# #### 数据预处理

# In[23]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape, type(x_train[0]))
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_train = (x_train - 127.5) / 127.5
buffer_size = 60000
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)
print(x_train.shape)


# #### 生成器模型

# In[27]:


from keras import layers

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256,use_bias=False,input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None,7,7,256)#注意：batch size没有限制
    
    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))
    assert model.output_shape == (None,7,7,128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
    assert model.output_shape == (None,14,14,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))
    assert model.output_shape == (None,28,28,1)
    
    return model


# In[28]:


generator = make_generator_model()
generator.summary()


# In[41]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap="gray")


# #### 判别器模型

# In[30]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# In[31]:


discriminator = make_discriminator_model()
discriminator.summary()
decision = discriminator(generated_image)
print(decision)


# #### 损失函数

# In[32]:


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    loss = real_loss + fake_loss
    return loss

def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


# #### 优化器

# In[33]:


generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)


# #### 超参数

# In[34]:


epochs = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# #### 定义训练过程

# In[35]:


@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size,noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output =discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss,disc_loss


# In[46]:


import time
from IPython import display

def train(dataset,epochs):
    for epoch in range(epochs):
        start=time.time()
        for i,image_batch in enumerate(dataset):
            g,d = train_step(image_batch)
            print("batch %d, gen_loss %f, disc_loss %f" % (i,g.numpy(),d.numpy()))

    #     display.clear_output(wait=True)
    #     generate_images(generator,epoch+1)
              
        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start))


# #### 显示生成图片

# In[43]:


def generate_images(model,test_input):
    predictions = model(test_input,training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:, :, 0]*127.5 + 127.5)
        plt.axis('off')
    plt.show()


# #### 训练

# In[44]:


get_ipython().run_cell_magic('time', '', 'train(train_dataset, epochs)')


# In[45]:


test_input = tf.random.normal([16, 100])
generate_images(generator, test_input)


# ## anime-faces

# In[26]:


import os
image_root = "data/anime-faces"
image_names = os.listdir(image_root)
print(len(image_names))
print(image_names[:10])


# In[22]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
img = plt.imread(os.path.join(image_root, image_names[0]))
print(img.shape, type(img))
plt.imshow(img)


# In[ ]:


from tqdm import tqdm
x = []
for name in tqdm(image_names):
    img = plt.imread(os.path.join(image_root, name))
    x.append(img)


# In[34]:


x = np.array(x)
x = (x - 127.5) / 127.5
buffer_size = 60000
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)


# In[ ]:


from keras import layers

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256,use_bias=False,input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None,7,7,256)#注意：batch size没有限制
    
    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))
    assert model.output_shape == (None,7,7,128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
    assert model.output_shape == (None,14,14,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))
    assert model.output_shape == (None,28,28,1)
    
    return model

