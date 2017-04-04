import os
from keras.layers import Dense, Activation
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    '''
    Method to load training, test and validation data from MNIST
    :return:
    '''

    # use tensorflow to load the data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_model():
    '''
    Method to create a new neural network model
    :return: the neural network
    '''

    # create a feedforward neural network
    model = Sequential()

    # add a fully connected layer to the input layer
    model.add(Dense(15, input_dim=784))

    # add the output layer
    model.add(Dense(10))

    # activate the output layer using softmax
    model.add(Activation('softmax'))

    return model


def save_model(model):
    '''
    Method to save the model and weights to the filesystem
    :param model:
    :return: boolean
    '''
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        print("Saved model to disk")
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved weights to disk")

        return True

    return False


def load_model():
    '''
    Method to load the model from the file system
    :return: the loaded model else None
    '''
    # load the model and weights from the path if it exists and return it
    if os.path.exists(os.path.join('.', 'model.json')):
        with open(os.path.join('.', 'model.json'), 'r') as model_file:
            json_model = model_file.read()
            model_net = model_from_json(json_model)
            model_net.load_weights('model.h5', by_name=True)

            print("Loaded model from file")
            return model_net

    return None


if __name__ == '__main__':

    # load the data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    # get the model and print summary
    model = get_model()
    model.summary()

    # train the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.01))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=128,
        nb_epoch=30,
        verbose=2
    )

    # save the model to the filesystem
    save_model(model)

    # evaluate the model on the test data and print metrics
    metrics = model.evaluate(X_test, y_test, batch_size=128, verbose=2)
    print("Evaluated model on validation data")
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i],
        print('{} {}'.format(metric_name, metric_value))
