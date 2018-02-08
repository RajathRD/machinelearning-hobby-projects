import tensorflow as tf
import scipy.stats as st
from scipy import misc
import numpy as np

def gkern(kernlen, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
it=int(input("Enter value"))
input_data = misc.imread('images/1.jpg')
height= input_data.shape[0]
width= input_data.shape[1]
image=tf.image.rgb_to_grayscale(input_data, name=None)

r = tf.reshape(input_data[...,0],[-1,height,width,1])
r = tf.to_float(r)
g = tf.reshape(input_data[...,1],[-1,height,width,1])
g = tf.to_float(g)
b = tf.reshape(input_data[...,2],[-1,height,width,1])
b = tf.to_float(b)

x = tf.reshape(image,[-1,height,width,1])
change_image = tf.to_float(x)
gauss_blur= gkern(it)
gauss_blur = tf.reshape(gauss_blur,[it,it,1,1])
gauss_blur=tf.to_float(gauss_blur)
img_r = tf.nn.conv2d(r, gauss_blur,strides=[1, 1, 1, 1], padding='SAME')
img_g = tf.nn.conv2d(g, gauss_blur,strides=[1, 1, 1, 1], padding='SAME')
img_b = tf.nn.conv2d(b, gauss_blur,strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
     sess.run(tf.initialize_all_variables())
     output = tf.reshape(img_r,[height,width])
     output = tf.reshape(img_g,[height,width])
     output = tf.reshape(img_b,[height,width])
     output_image = np.array(output.eval(), dtype='uint8')
     misc.imshow(input_data)
     misc.imshow(output_image)
