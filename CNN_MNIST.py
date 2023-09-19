
# imports
from __future__ import print_function
import numpy as np
import keras
from keras.utils import np_utils
from keras import utils as np_utils
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.regularizers import l2
import time 
from sklearn.metrics import ConfusionMatrixDisplay
import sklearn.metrics
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

#%%  
# Hyper-parameters data-loading and formatting

batch_size = 128
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28

(x_train, lbl_train), (x_test, lbl_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
   
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.np_utils.to_categorical(lbl_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(lbl_test, num_classes)


#%%
# best - 32 (3,3), 128 (3,3), 512 (3,3)
t = time.time()
## Define model ##
model = Sequential()

model.add(Conv2D(32, (7,7), padding="same", activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (7,7), padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(512, (9,9), activation = 'linear'))


model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=tensorflow.keras.optimizers.SGD(learning_rate = 0.2),
        metrics=[tf.keras.metrics.Precision()],)

fit_info = model.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


print( 'time =', time.time() - t ) 
print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))


#%%
y_pred = model.predict(x_test)
y = np.argmax(y_test,  axis=1)
confus =  sklearn.metrics.confusion_matrix(np.argmax(y_test,  axis=1), np.argmax(y_pred, axis=1))
print(confus)
print(classification_report(y_test, np.round(y_pred), digits = 5))


miss_classification = 1 - precision_recall_fscore_support(y_test, np.round(y_pred),average = 'micro')[0]
print('Miss-clasification rate =',miss_classification)
#model.summary()

#%%
# summarize history for accuracy
plt.plot(fit_info.history['accuracy'])
plt.plot(fit_info.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()