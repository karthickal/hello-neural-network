import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def display_image(image, colormap=None):
    plt.imshow(image, cmap=colormap)
    plt.show()


def display_image_grid(images, per_row=4, colormap=None, fsize=(40, 20)):
    '''
    Utility method to show the images in a grid
    :param images: the input image
    :param per_row: number of images per row
    :param colormap: colormap for the images
    :param fsize: figure size
    :return:
    '''
    num_rows = math.ceil(len(images) / per_row)

    fig = plt.figure(figsize=fsize)

    grid = gridspec.GridSpec(num_rows, per_row, wspace=0.0)
    ax = [plt.subplot(grid[i]) for i in range(num_rows * per_row)]
    fig.tight_layout()

    for i, img in enumerate(images):
        ax[i].imshow(img, cmap=colormap)
        ax[i].axis('off')

    plt.show()


def convert_image(x):
    '''
    Method to convert a numpy array to a valid image structure
    :param x: the numpy array
    :return: numpy array reshaped
    '''
    return x.reshape(28, 28)


def load_random_data(X, y):
    '''
    Method to load random data from the given dataset
    :param X: images
    :param y: labels
    :return: a random X, y
    '''
    ind = np.random.randint(0, len(X))
    return X[ind], y[ind]
