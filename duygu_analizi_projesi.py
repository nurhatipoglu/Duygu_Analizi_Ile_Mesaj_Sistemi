# -*- coding: utf-8 -*-
"""Duygu Analizi Projesi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XaFQG-nxopkeeuu8mRRgoXLhewnSPh1L
"""

import sys,os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D, MaxPool2D
from keras.losses import categorical_crossentropy  
from keras.optimizers import Adam  
from keras.regularizers import l2 
from keras.utils import np_utils
from keras.constraints import max_norm

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive

df=pd.read_csv('/gdrive/My Drive/Proje/fer2013.csv')
print(df.head)

X_train,train_y,X_test,test_y=[],[],[],[]  
for index, row in df.iterrows():  
    val=row['pixels'].split(" ")  
    if 'Training' in row['Usage']:
      X_train.append(np.array(val,'float32'))  
      train_y.append(row['emotion'])  
    elif 'PublicTest' and 'PrivateTest' in row['Usage']:  
      X_test.append(np.array(val,'float32'))  
      test_y.append(row['emotion'])

num_features = 64  
num_labels = 7  
batch_size = 64  
epochs = 50
#sayıları float tipi arraye dönüştür
X_train = np.array(X_train,'float32')  
X_test = np.array(X_test,'float32') 
X_train.shape
train_y = keras.utils.to_categorical(train_y, 7)#7 kategoriye göre matris yapar
test_y = keras.utils.to_categorical(test_y, 7)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)  
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
input_shape = (48 , 48, 1)

X_train /= 255 #verileri normalleştir
X_test /=255

max_norm_value = 2.0
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
os.chdir('/gdrive/My Drive/Proje')
checkpointer = ModelCheckpoint(filepath='/face_model.h5', verbose=1, save_best_only=True)

# Fit data to model
hist = model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpointer],
          validation_split=0.2
)

model.save_weights("face_model.h5")

scores = model.evaluate(X_test, test_y, verbose=1)
print(100*scores[1])
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

os.chdir('/gdrive/My Drive/Proje')
m_json = model.to_json()  
with open("face_model.json", "w") as json_file:  
    json_file.write(m_json)

from google.colab import files
files.download("face_model.json")
files.download("face_model.h5")

# Commented out IPython magic to ensure Python compatibility.
from matplotlib import pyplot as plt
# %matplotlib inline
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Eğitim', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist.history['loss'], color ='r', label='Training Loss')
plt.plot(hist.history['val_loss'], color ='b', label='Validation Loss')
plt.legend(loc='upper right')


plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(hist.history['accuracy'], color ='g', label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], color ='m', label='Validation Accuracy')
plt.legend(loc='lower right')

plt.show()

plt.suptitle('Model Accurarcy', fontsize=10)
plt.plot(hist.history['accuracy'], color ='g', label='Training Accuracy')
plt.ylabel('Accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.legend(['train'],loc='upper left')
plt.show()

scores = model.evaluate(X_test, test_y, verbose=1)
print(100*scores[1])
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])