import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, MaxPooling2D, Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os


# Build the basic model
def build_basic_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape, name='Conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MP1'))

    model.add(Conv2D(32, 5, activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='MP2'))

    model.add(Flatten(name='Flatten'))
    model.add(Dense(128, activation='relu', name='Dense1'))
    model.add(Dense(num_classes, activation='softmax', name='Dense2'))
    return model


# Build the intermediate-I model
def build_intermediate_model1(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape, name='Conv1'))
    model.add(BatchNormalization(name='BN1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MP1'))

    model.add(Conv2D(32, 5, activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='BN2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MP2'))

    model.add(Flatten(name='Flatten'))
    model.add(Dense(128, activation='relu', name='Dense1'))
    model.add(BatchNormalization(name='BN3'))
    model.add(Dense(num_classes, activation='softmax', name='Dense2'))
    return model


# Build the intermediate-II model
def build_intermediate_model2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape, name='Conv1'))
    model.add(BatchNormalization(name='BN1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MP1'))
    model.add(Dropout(0.25, name='Dropout1'))

    model.add(Conv2D(32, 5, activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='BN2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MP2'))
    model.add(Dropout(0.25, name='Dropout2'))

    model.add(Flatten(name='Flatten'))
    model.add(Dense(128, activation='relu', name='Dense1'))
    model.add(BatchNormalization(name='BN3'))
    model.add(Dropout(0.5, name='Dropout3'))
    model.add(Dense(num_classes, activation='softmax', name='Dense2'))
    return model


# Build the advanced model
def build_advanced_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape, name='Conv1'))
    model.add(BatchNormalization(name='BN1'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='BN2'))
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu', name='Conv3'))
    model.add(BatchNormalization(name='BN3'))
    model.add(Dropout(0.4, name='Dropout1'))

    model.add(Conv2D(64, kernel_size=3, activation='relu', name='Conv4'))
    model.add(BatchNormalization(name='BN4'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', name='Conv5'))
    model.add(BatchNormalization(name='BN5'))
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu', name='Conv6'))
    model.add(BatchNormalization(name='BN6'))
    model.add(Dropout(0.4, name='Dropout2'))

    model.add(Flatten(name='Flatten1'))
    model.add(Dense(128, activation='relu', name='Dense1'))
    model.add(BatchNormalization(name='BN7'))
    model.add(Dropout(0.4, name='Dropout3'))
    model.add(Dense(num_classes, activation='softmax', name='Dense2'))
    return model


def get_callbacks_list(model_name, add_checkpoint: bool = False):
    schdler = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=1)
    callbacks_list = [schdler]
    if add_checkpoint:
        strTmp = 'checkpoint\\{}_model'.format(model_name)
        checkpoint_filepath = strTmp + r'_{epoch:02d}_{accuracy:.2f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list.append(checkpoint)
    return callbacks_list


def plot_accuracy(history, labels, title):
    styles = [':', '-.', '--', '-']
    n = len(history)
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(ylim=(0.97, 1.0), xlabel='Epoch', ylabel='Validation accuracy')
    for i in range(n):
        plt.plot(history[i].history['val_accuracy'], linestyle=styles[i])
    plt.title(title)
    plt.legend(labels, loc='upper left')
    plt.show()


def main():
    print("Started to build the basic, intermediate and advanced models")

    gpu_available = tf.test.is_gpu_available()
    if gpu_available:
        tf.device("/gpu:0")
    else:
        tf.device("/cpu:0")
    np.random.seed(555)

    # -------------------------- load data -----------------------------
    input_shape = (28, 28, 1)  # 'channel_last'
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)
    x_train = x_train.reshape(-1, *input_shape).astype("float32") / 255
    x_test  = x_test.reshape(-1, *input_shape).astype("float32") / 255
    print(x_train.shape, y_train.shape)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(x_train.shape, y_train.shape)
    # split the dataset into two subsets: one for training and the other for validation ------
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, test_size=1/3)

    batch_size = 64
    epochs = 35

    num_models = 4
    history = [0] * num_models
    test_scores = []
    model_names = ["Basic", "Intermediate1", "Intermediate2", "Advanced"]

    models = [None] * num_models
    models[0] = build_basic_model(input_shape, num_classes)
    models[1] = build_intermediate_model1(input_shape, num_classes)
    models[2] = build_intermediate_model2(input_shape, num_classes)
    models[3] = build_advanced_model(input_shape, num_classes)

    # train the models one by one
    for i in range(num_models):

        callbacks_list = get_callbacks_list(model_names[i], add_checkpoint=False)

        save_model_fname = r'model\mnist_{}_model_batchsize{}_epoch{}.h5'.format(model_names[i], batch_size, epochs)
        if os.path.isfile(save_model_fname):  # previously trained
            models[i] = load_model(save_model_fname)
        else:  # new training
            models[i].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # print model summary information
            print(models[i].summary())

            # training this model
            start_time = time.time()
            if i < num_models - 1:
                history[i] = models[i].fit(X_train2, Y_train2,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           validation_data=(X_val2, Y_val2),
                                           callbacks=callbacks_list,
                                           verbose=0)
            else:
                # using Data Augmentation
                datagen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             zoom_range=0.1,
                                             horizontal_flip=False,
                                             vertical_flip=False)

                history[i] = models[i].fit(datagen.flow(X_train2, Y_train2, batch_size=batch_size),
                                           epochs=epochs,
                                           validation_data=(X_val2, Y_val2),
                                           callbacks=callbacks_list,
                                           verbose=0)

            end_time = time.time()
            models[i].save(save_model_fname)
            print('{} model: Epochs={:d}, Train accuracy={:.5f}, Validation accuracy={:.5f}, training time {:.0f} seconds'
                  .format(model_names[i], epochs, max(history[i].history['accuracy']),
                          max(history[i].history['val_accuracy']), end_time - start_time))

        test_score = models[i].evaluate(x_test, y_test, verbose=0)
        test_scores.append(test_score)

        print('{} model: Epochs={:d}, Test loss: {}, Test accuracy: {}'
              .format(model_names[i], epochs, test_scores[i][0], test_scores[i][1]))

    plot_accuracy(history, model_names, 'Comparison of validation accuracy of models')


if __name__ == '__main__':
    main()