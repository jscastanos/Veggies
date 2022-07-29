import os
import random

import cv2
import numpy as np
import pandas as pd

DATADIR = "./images"


def getCategories():
    # image directories are categories
    dirs = []

    for dir in os.listdir(DATADIR):
        if os.path.isdir(os.path.join(DATADIR, dir)):
            dirs.append(dir)
    return dirs


def getTrainingData(image_size, categories):
    data = []

    for category in categories:
        # path for data dir and categories
        path = os.path.join(DATADIR, category)
        label = categories.index(category)

        for img in os.listdir(path):
            # read image
            file = os.path.join(path, img)
            # convert to grayscale to reduce data size
            grayImage = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # resize image to have the same dimension
            newImage = cv2.resize(grayImage, (image_size, image_size))
            # add to array
            data.append([newImage, label])
    return data


categories = getCategories()
training_data = getTrainingData(250, categories)

# shuffle to avoid bias
random.shuffle(training_data)

# convert to np array
training_data = np.array(training_data, dtype="object")

# split array
X_train = training_data.T[0]
y_train = training_data.T[1]


# save dataset to file
np.savez("./veggies-dataset.npz", X_train=X_train, y_train=y_train)

# save categories to file
df = pd.DataFrame(categories, columns=['category'])
df.to_csv('veggies-category.csv', index=False)
