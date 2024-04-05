#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers,models


# In[2]:


def create_convolutional_layer(filters,kernel_size,activation='relu',input_shape=None):
    if(input_shape):
        return layers.Conv2D(filters,kernel_size,activation=activation,input_shape=input_shape)
    else:
        return layers.Conv2D(filters,kernel_size,activation=activation)


# In[3]:


def create_maxpooling_layer(pool_size=(2,2)):
    return layers.MaxPooling2D(pool_size)


# In[4]:


def create_dense_layer(units,activation='relu'):
    return layers.Dense(units,activation=activation)


# In[8]:


def build_convnet(input_shape,num_classes):
    model=models.Sequential()
    model.add(create_convolutional_layer(32,(3,3),input_shape=input_shape))
    model.add(create_maxpooling_layer())
    model.add(create_convolutional_layer(64,(3,3)))
    model.add(create_maxpooling_layer())
    model.add(layers.Flatten())
    model.add(create_dense_layer(128))
    model.add(create_dense_layer(num_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model


# In[9]:


(train_images,train_labels),(test_images,test_labels)=tf.keras.datasets.fashion_mnist.load_data()


# In[10]:


train_images=train_images.reshape((60000,28,28,1)).astype('float32')/255
test_images=test_images.reshape((10000,28,28,1)).astype('float32')/255
input_shape=(28,28,1)
num_classes=10
model=build_convnet(input_shape,num_classes)
model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))


# In[ ]:




