import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

def load_data(image_path, label_path, num_classes=10, normalize=True):
    """Load and preprocess dataset."""
    images = np.load(image_path)
    labels = np.load(label_path)

    # Normalize images
    if normalize:
        mean = np.mean(images, axis=(0, 1, 2, 3))
        std = np.std(images, axis=(0, 1, 2, 3))
        images = (images - mean) / (std + 1e-7)

    # One-hot encode labels
    labels = np_utils.to_categorical(labels, num_classes)

    return images, labels

def augment_data():
    """Define data augmentation pipeline."""
    return ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
