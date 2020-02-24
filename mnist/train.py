import os

import tensorflow as tf

from model import get_model
from dataset import dataset


def main():
    """
    Get the dataset, model
    Set the callback
    Train and save the best weights based on validation accuracy

    """
    train_images, train_labels, test_images, test_labels = dataset()

    model = get_model()

    model.summary()

    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    model.fit(train_images,
              train_labels,
              epochs=100,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback],
              verbose=2)


if __name__ == '__main__':
    main()
