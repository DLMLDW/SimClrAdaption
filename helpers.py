'''
Adapted from: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2
'''


import tensorflow as tf
import numpy as np

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
	"""Blurs the given image with separable convolution.
	Args:
	  image: Tensor of shape [height, width, channels] and dtype float to blur.
	  kernel_size: Integer Tensor for the size of the blur kernel. This is should
		be an odd number. If it is an even number, the actual kernel size will be
		size + 1.
	  sigma: Sigma value for gaussian operator.
	  padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
	Returns:
	  A Tensor representing the blurred image.
	"""
	radius = tf.to_int32(kernel_size / 2)
	kernel_size = radius * 2 + 1
	x = tf.to_float(tf.range(-radius, radius + 1))
	blur_filter = tf.exp(
		-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.to_float(sigma), 2.0)))
	blur_filter /= tf.reduce_sum(blur_filter)
	# One vertical and one horizontal filter.
	blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
	blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
	num_channels = tf.shape(image)[-1]
	blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
	blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
	expand_batch_dim = image.shape.ndims == 3
	if expand_batch_dim:
	  # Tensorflow requires batched input to convolutions, which we can fake with
	  # an extra dimension.
	  image = tf.expand_dims(image, axis=0)
	blurred = tf.nn.depthwise_conv2d(
		image, blur_h, strides=[1, 1, 1, 1], padding=padding)
	blurred = tf.nn.depthwise_conv2d(
		blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
	if expand_batch_dim:
	  blurred = tf.squeeze(blurred, axis=0)
	return blurred



def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


def gaussian_filter(v1, v2):
    k_size = int(v1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width
    gaussian_ope = gaussian_blur(kernel_size=k_size, min=0.1, max=2.0)
    [v1, ] = tf.py_function(gaussian_ope, [v1], [tf.float32])
    [v2, ] = tf.py_function(gaussian_ope, [v2], [tf.float32])
    return v1, v2
