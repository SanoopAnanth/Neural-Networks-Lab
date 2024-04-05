#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb


# In[2]:


max_features=10000
max_len=500
batch_size=32


# In[3]:


(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)


# In[4]:


print(len(input_train),'train_sequences')
print(len(input_test),'test sequences')


# In[10]:


input_train=sequence.pad_sequences(input_train,maxlen=max_len)
input_test=sequence.pad_sequences(input_test,maxlen=max_len)
print('input_train shape',input_train.shape)
print('input_test shape',input_test.shape)


# In[11]:


model=Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(input_train,y_train,epochs=10,batch_size=batch_size,validation_split=0.2)
score,acc=model.evaluate(input_test,y_test,batch_size=batch_size)


# In[12]:


print('test score:',score)
print('test accuracy:',acc)


# In[ ]:




