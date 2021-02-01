# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:09:03 2020
Use convolution to build GAN and save the model as 'CovGanNet_17.h5', and the splicing data is 'X_train_cov_17'
@author: Administrator
"""
from keras.layers import Dense,Input,MaxPooling2D, Flatten, Conv2D
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import numpy as np


"""Select GPU"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size=256
epoch_number=100000

## The size of a sliding window 
soildnumber=17
##Import real protein data
with np.load('win17_train.npz') as data_train:
    x1 =  data_train['X_train']
x_train1 = []
x_train = []
for i in range (len(x1)):
    x_train1.append(x1[i].reshape(soildnumber,20,1))
    x_train.append(x_train1[i].transpose(1,0,2))
x_train = np.array(x_train)



def build_generator():
  model=Sequential()
  model.add(Conv2D(64, 2, strides=1, padding='same', input_shape=(20,soildnumber,1), activation='relu'))
  model.add(Conv2D(64, 2, strides=1, padding='same',activation='relu'))
  model.add(Conv2D(1, 2, strides=1, padding='same',activation='relu'))
  g_adam = Adam(0.0005,0.5)
  model.compile(loss='binary_crossentropy',optimizer=g_adam) 
  return model
 
generator=build_generator()
generator.build((None,20,soildnumber,1))

def build_discriminator():
  model=Sequential()                                       
  model.add(Conv2D(64, 2, strides=1, padding='same', input_shape=(20,soildnumber,1),activation='relu'))
  model.add(MaxPooling2D(2, 1, padding='same'))
  model.add(Conv2D(64, 2, strides=1, padding='same',activation='relu'))
  model.add(MaxPooling2D(2, 1, padding='same'))
  model.add(Flatten())
  model.add(Dense(units=1,activation='sigmoid'))
  d_adam = Adam(0.0005,0.5)
  model.compile(loss='binary_crossentropy',optimizer=d_adam)
  model.build((None,soildnumber,20))

  return model
 
discriminator=build_discriminator()



def build_GAN(discriminator,generator): 
  discriminator.trainable=False
  GAN_input=Input(shape=(20,soildnumber,1))
  x=generator(GAN_input)
  GAN_output=discriminator(x)
  GAN=Model(inputs=GAN_input,outputs=GAN_output)
  GAN_rmsprop = RMSprop(lr=GAN_learnirate, rho=0.9, epsilon=1e-08, decay=0.005)

  GAN.compile(loss='binary_crossentropy',optimizer=GAN_rmsprop)
  return GAN
 
GAN=build_GAN(discriminator,generator)

 
def train_GAN(epochs, batch_size):
    # loading the data

    generator = build_generator()
    discriminator = build_discriminator()
    GAN = build_GAN(discriminator, generator)

    for i in range(1, epochs + 1):
        print("Epoch %d" % i)

        noise = np.random.randint(-9, 9, size=(batch_size, 20,soildnumber,1), dtype='int16')
        # Generate fake image (data)
        fake_images = generator.predict(noise)

        # Take the real data of random batch
        real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
       
        #  Establish true and fake data labels
        label_fake = np.zeros(batch_size)
        label_real = np.ones(batch_size)
        for j in range(batch_size):
           label_real[j] = np.array([0.9])

        x = np.concatenate([fake_images, real_images])
        y = np.concatenate([label_fake, label_real])

        # Training discriminator
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x, y)

        # Training generator
        discriminator.trainable = False
        g_loss = GAN.train_on_batch(noise, label_real)

        if i % 5 == 0:
            print('discriminator loss at step %s: %s' % (i, d_loss))
            print('generator loss at step %s: %s' % (i, g_loss))

#Start training
train_GAN(epochs=epoch_number,batch_size=batch_size)


#Setting generator weights is not trainable

generator.trainable=False
generator.save('CovGanNet_13.h5')

z1=generator.predict(x_train)
z2=z1.reshape(-1,20,soildnumber)

#The generated features are spliced with PSSM
conall=[]
j=0
for j in range(len(x1)):
    con_one=np.concatenate((x1[j],z2[j]),axis=0)
    conall.append(con_one)
    
x_con=np.array(conall)
np.save('X_train_cov_17', x_con)