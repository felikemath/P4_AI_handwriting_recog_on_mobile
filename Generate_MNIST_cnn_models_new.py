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


# BUild a basic model
def build_basic_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape, name='1st_layer_Conv2D'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='myMP1'))

    model.add(Conv2D(32, 5, activation='relu', name='myconv2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='myMP2'))

    model.add(Flatten(name='myFlatten'))
    model.add(Dense(128, activation='relu', name='myDense1'))
    model.add(Dense(num_classes, activation='softmax', name='myDense2'))
    return model


# BUild an intermediate model
def build_intermediate_model1(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape, name='myConv1'))
    model.add(BatchNormalization(name='myBN1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='myMP1'))

    model.add(Conv2D(32, 5, activation='relu', name='myConv2'))
    model.add(BatchNormalization(name='myBN2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='myMP2'))

    model.add(Flatten(name='myFlatten'))
    model.add(Dense(128, activation='relu', name='myDense1'))
    model.add(BatchNormalization(name='myBN3'))
    model.add(Dense(num_classes, activation='softmax', name='myDense2'))
    return model


def build_intermediate_model2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape, name='myConv1'))
    model.add(BatchNormalization(name='myBN1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='myMP1'))
    model.add(Dropout(0.25, name='myDropout1'))

    model.add(Conv2D(32, 5, activation='relu', name='myConv2'))
    model.add(BatchNormalization(name='myBN2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='myMP2'))
    model.add(Dropout(0.25, name='myDropout2'))

    model.add(Flatten(name='myFlatten'))
    model.add(Dense(128, activation='relu', name='myDense1'))
    model.add(BatchNormalization(name='myBN3'))
    model.add(Dropout(0.5, name='myDropout3'))
    model.add(Dense(num_classes, activation='softmax', name='myDense2'))
    return model


# BUild an advanced model
def build_advanced_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape, name='myConv1'))  # 32
    model.add(BatchNormalization(name='myBN1'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', name='myConv2'))
    model.add(BatchNormalization(name='myBN2'))
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu', name='myConv3'))
    model.add(BatchNormalization(name='myBN3'))
    model.add(Dropout(0.4, name='myDropout1'))

    model.add(Conv2D(64, kernel_size=3, activation='relu', name='myConv4'))
    model.add(BatchNormalization(name='myBN4'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', name='myConv5'))
    model.add(BatchNormalization(name='myBN5'))
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu', name='myConv6'))
    model.add(BatchNormalization(name='myBN6'))
    model.add(Dropout(0.4, name='myDropout2'))

    model.add(Flatten(name='myFlatten1'))
    model.add(Dense(128, activation='relu', name='myDense1'))
    model.add(BatchNormalization(name='myBN7'))
    model.add(Dropout(0.4, name='myDropout3'))
    model.add(Dense(num_classes, activation='softmax', name='myDense2'))
    return model


def get_callbacks_list(add_checpoint: bool = False):
    schdler = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
    callbacks_list = [schdler]
    if add_checpoint:
        checkpoint_filepath=r'checkpoint\mnist3_weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list.append(checkpoint)
    return callbacks_list


def main():
    print("Started generating the basic, intermediate and advanced MNIST models")

    tf.device("/gpu:0")  # "/cpu:0"
    np.random.seed(1234)

    # -------------------------- load data -----------------------------
    input_shape = (28, 28, 1)  # K.image_data_format() = 'channels_last'
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    # plt.imshow(x_train[0], cmap='gray')
    x_train = x_train.reshape(-1, *input_shape).astype("float32") / 255
    x_test  = x_test.reshape(-1, *input_shape).astype("float32") / 255
    print(x_train.shape)
    print(y_train.shape)
    # print(y_train[:10])
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(x_train.shape, y_train.shape)
    # CREATE VALIDATION SET ------
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, test_size=1/3)

    # Model training
    batch_size = 64
    epochs = 35
    callbacks_list = get_callbacks_list()

    num_models = 4
    history = [0] * num_models
    test_scores = []
    model_names = ["Basic", "Intermediate1", "Intermediate2", "Advanced"]

    models = [None] * num_models
    models[0] = build_basic_model(input_shape, num_classes)
    models[1] = build_intermediate_model1(input_shape, num_classes)
    models[2] = build_intermediate_model2(input_shape, num_classes)
    models[3] = build_advanced_model(input_shape, num_classes)

    for i in range(num_models):

        save_model_fname = r'model\mnist_{}_model_batchsize{}_epoch{}.h5'.format(model_names[i], batch_size, epochs)
        if os.path.isfile(save_model_fname):
            models[i] = load_model(r'.\model\model_mnist_expt5_test4_batchsize64_epoch35_BN_DO_DA.h5')
        else:
            # configure the learning process by doing compile the model
            models[i].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # print model summary
            print(models[i].summary())

            # training the models
            start_time = time.time()
            if i < num_models - 1:
                history[i] = models[i].fit(X_train2, Y_train2,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           validation_data=(X_val2, Y_val2),
                                           callbacks=callbacks_list,
                                           verbose=0)
            else:
                # Data augmentation
                datagen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             zoom_range=0.1,
                                             horizontal_flip=False,
                                             vertical_flip=False)

                # fit_generator
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

            # plt.plot(history[i].history['accuracy'])
            # plt.show(block=False)

        test_score = models[i].evaluate(x_test, y_test, verbose=0)
        test_scores.append(test_score)

        print('{} model: Epochs={:d}, Test loss: {}, Test accuracy: {}'
              .format(model_names[i], epochs, test_scores[i][0], test_scores[i][1]))





    # PLOT ACCURACIES
    styles = [':', '-.', '--', '-']
    fig= plt.figure(figsize=(15, 10))
    ax = plt.axes(ylim=(0.97, 1.0), xlabel='Epoch', ylabel='Validation accuracy')
    for i in range(num_models):
        plt.plot(history[i].history['val_accuracy'], linestyle=styles[i])
    plt.title('Comparison of valiation accuracy of models')
    plt.legend(model_names, loc='upper left')
    plt.show()

    print('Done')


if __name__ == '__main__':
    main()