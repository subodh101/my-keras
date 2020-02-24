import os

import tensorflow as tf

from dataset import dataset
from model import get_model


def main():
    """
    Get the test dataset and model
    Load the best weights to the model
    Predict the result from the test dataset

    Returns:

    """
    _, _, test_images, test_labels = dataset()

    model = get_model()

    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model.load_weights(latest)

    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    main()
