import train
import utils
import numpy as np
import matplotlib.pyplot as plt


def display_samples(X_data, y_data, count=12):
    '''
    Display sample images and labels from the given dataset
    :param X_data: images
    :param y_data: labels
    :param count: count of data to be displayed
    :return: None
    '''
    data = []
    labels = []
    for i in range(count):
        x, y = utils.load_random_data(X_data, y_data)
        data.append(utils.convert_image(x))
        labels.append(np.argmax(y))

    print('Labels for displayed digits: ', labels)
    utils.display_image_grid(data, colormap='gray')


def display_digit_samples(X_data, y_data, digit=1, count=12):
    '''
    Display sample images of a specific digit
    :param X_data: images
    :param y_data: labels
    :param digit: the digit required
    :param count: count of data to be displayed
    :return: None
    '''
    data = []
    labels = []

    while count != 0:
        x, y = utils.load_random_data(X_data, y_data)
        dig = np.argmax(y)
        if dig != digit:
            continue

        data.append(utils.convert_image(x))
        labels.append(dig)
        count = count - 1

    print('Labels for displayed digits: ', labels)
    utils.display_image_grid(data, colormap='gray')


def display_single(X_data, y_data):
    '''
    Method to display a single data
    :param X_data: image
    :param y_data: label
    :return: None
    '''
    x, y = utils.load_random_data(X_data, y_data)
    img = x.reshape(28, 28)
    print('Label Data: ', y)

    utils.display_image(img, colormap='gray')


def display_histogram(y_data):
    '''
    Method to display the frequency of the data
    :param y_data: labels
    :return: None
    '''
    y_labels = np.argmax(y_data, axis=1)
    print(np.arange(y_labels.min(), y_labels.max() + 2, 1))
    plt.hist(y_labels, bins=np.arange(y_labels.min(), y_labels.max() + 2, 1) - 0.5, rwidth=0.5)

    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = train.load_data()
    display_digit_samples(X_train, y_train, digit=1)
    display_single(X_train, y_train)
    display_samples(X_train, y_train)
    display_histogram(y_train)
