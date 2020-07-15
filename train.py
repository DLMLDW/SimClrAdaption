'''
Author: Dominik Waibel
Adapted from: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2
'''

from losses import *
from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
import helpers
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
import numpy as np
from Augmentations import CustomAugment
import logging

def negativemask(BATCH_SIZE):
    negative_mask = helpers.get_negative_mask(BATCH_SIZE)
    return negative_mask

@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature, BATCH_SIZE):
    negative_mask = negativemask(BATCH_SIZE)
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_simclr(model, dataset, optimizer, criterion, BATCH_SIZE, epochs,
                 temperature=0.1):
    data_augmentation = Sequential([Lambda(CustomAugment())])
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in (range(epochs)):
        for image_batch in dataset:
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature, BATCH_SIZE)
            step_wise_loss.append(loss)
            logging.info({"epoch loss": np.mean(loss)})
        epoch_wise_loss.append(np.mean(step_wise_loss))
        logging.info({"nt_xentloss": np.mean(step_wise_loss)})

        if epoch % 10 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, model