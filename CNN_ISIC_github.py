# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:47:18 2023

@author: lulle
"""


import numpy as np
import pandas as pd


from keras import backend as K
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import time
from sklearn.model_selection import train_test_split
import sklearn.metrics
from collections import Counter


#%% Import datafiles 


x_train = pd.read_csv(r'XXXX\Xdata_train_example_case.csv')
y_train = pd.read_csv(r'XXXX\Ydata_train_example_case.csv')

x_test = pd.read_csv(r'XXXX\Xdata_test_example_case.csv')
y_test = pd.read_csv(r'XXXX\Ydata_test_example_case.csv')


#%% data-loading and formatting

img_rows, img_cols = 100, 150

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

#%%  USED FOR ENTIRE DATASET, NOT SAMPLE

# re-sample to fix imbalance in training data


# from random import choices, shuffle

# print(Counter(y_train))

# cancer = np.where(y_train == 1)[0]
# non_cancer = list(range(len(y_train)))

# for i in cancer:
#     non_cancer.remove(i)


# sample_cancer = choices(cancer, k=int(len(y_train)/2))
# sample_non_cancer = choices(non_cancer, k=int(len(y_train)/2))

# sample = sample_non_cancer + sample_cancer
# shuffle(sample)

# x_train_balanced = x_train[sample]
# y_train_balanced = y_train[sample].reshape(-1)

# print(Counter(y_train_balanced))

# %%  CNN 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 64
epochs = 10

t = time.time()
model = Sequential()
model.add(Conv2D(32, 4, activation = 'relu',
                 input_shape = (img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(64, 4, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

#model.save_weights('inital.h5')

fit_info = model.fit(x_train_balanced, y_train_balanced,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print()
print( 'time = ', (time.time() - t)/60)
print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))


#%% Prediction of test data 

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


y_pred = (model.predict(x_test) > 0.5).astype("int32")


confus =  sklearn.metrics.confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
print(confus)
print(classification_report(y_test, np.round(y_pred), digits = 5))

miss_classification = 1 - precision_recall_fscore_support(y_test, np.round(y_pred),average = 'micro')[0]
print('Miss-clasification rate =', miss_classification)

#%% summarize history for accuracy

plt.plot(fit_info.history['accuracy'])

plt.plot(fit_info.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy rate')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
