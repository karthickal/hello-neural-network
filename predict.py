import train
import utils
import numpy as np

# count of data to be predicted
DATA_COUNT = 12

if __name__ == '__main__':
    model = train.load_model()
    X_train, y_train, X_test, y_test, X_val, y_val = train.load_data()

    # load some random data from the test set
    data = []
    images = []
    orig_labels = []
    for i in range(DATA_COUNT):
        x, y = utils.load_random_data(X_test, y_test)
        image = utils.convert_image(x)
        data.append(x)
        images.append(image)
        orig_labels.append(np.argmax(y))

    # get the softmax score for the data
    preds = model.predict(np.array(data), batch_size=DATA_COUNT)
    labels = np.argmax(preds, axis=1)

    print('Original Labels: ', orig_labels)
    print('Predicted Labels: ', labels)
    utils.display_image_grid(images, colormap='gray')
