'''
Author: Dominik Waibel
Adapted from: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2
'''


from Augmentations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from model import get_resnet_simclr
from train import train_simclr
import datetime
print(tf.__version__)
import os

Batch_size = 1
#Resize images during import to these dimensions
Image_dimensions = (512,512,3)
#Number of epochs to train
epochs = 30
#Specify the path to the images here:
train_images = list(glob.glob(os.getcwd() +"/Allimages_small/*"))
print(len(train_images))

data_augmentation = Sequential([Lambda(CustomAugment())])

# Image preprocessing utils
@tf.function
def parse_images(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[Image_dimensions[0], Image_dimensions[1]])

    return image


# Create TensorFlow dataset
train_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_ds = (
    train_ds
    .map(parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .shuffle(1024)
    .batch(Batch_size, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.SUM)
decay_steps = 1000
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

resnet_simclr_2 = get_resnet_simclr(256, 128, 50, Image_dimensions)

epoch_wise_loss, resnet_simclr  = train_simclr(resnet_simclr_2, train_ds, optimizer, criterion, Batch_size, epochs,
                 temperature=0.1)

with plt.xkcd():
    plt.plot(epoch_wise_loss)
    plt.title("tau = 0.1, h1 = 256, h2 = 128, h3 = 50")
    plt.show()

#save weights
filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "resnet_simclr.h5"

resnet_simclr.save_weights(filename)
