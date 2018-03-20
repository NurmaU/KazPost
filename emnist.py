import os
import numpy as np
from mnist import MNIST
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

batch_size = 128
num_classes = 26
epochs = 12
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def load_data(train_test):
    path = os.path.join('dataset', 'gzip')

    mndata = MNIST(path)
    if train_test == 'train':
        x_out, y_out = mndata.load_training()
    else:
        x_out, y_out = mndata.load_testing()

    letters = 'abcdefghijklmnopqrstuvwxyz'

    label_set = list()
    for el in letters:
        label_set.append(el)


    return x_out, list(np.array(y_out)-1)


def preprocessing():
    x_train, y_train = load_data('train')
    x_test, y_test = load_data('test')

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def model_def():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


def train_model(model):
    x_train, x_test, y_train, y_test = preprocessing()

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    model = model_def()
    train_model(model)

