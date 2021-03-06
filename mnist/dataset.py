import tensorflow as tf


def dataset():
    """
    Get an example dataset

    Returns: tuple(train_images, train_labels, test_images, test_labels)

    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    return train_images, train_labels, test_images, test_labels
