from __future__ import print_function

import os
import numpy as np
from keras.layers import Input, Reshape, Dense, Dropout, Activation, Flatten, PReLU, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Model
from keras import backend as K
from tqdm import tqdm
from PIL import Image

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

K.set_image_dim_ordering('tf')

# generator
nch = 256
g_input = Input(shape=(100, ))
g_input_cond = Input(shape=(10, ))
g = Dense(nch * 7 * 7, init='glorot_normal')(merge([g_input, g_input_cond], mode='concat', concat_axis=1))
g = BatchNormalization(mode=2)(g)
g = PReLU()(g)
g = Reshape([7, 7, nch])(g)
g = Convolution2D(nch, 3, 3, border_mode='same', init='glorot_uniform')(g)
g = BatchNormalization(axis=3, mode=2)(g)
g = PReLU()(g)
g = UpSampling2D(size=(2, 2))(g)
g = Convolution2D(nch / 2, 3, 3, border_mode='same', init='glorot_uniform')(g)
g = BatchNormalization(axis=3, mode=2)(g)
g = PReLU()(g)
g = UpSampling2D(size=(2, 2))(g)
g = Convolution2D(nch / 4, 3, 3, border_mode='same', init='glorot_uniform')(g)
g = BatchNormalization(axis=3, mode=2)(g)
g = PReLU()(g)
g = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(g)
g_output = Activation('sigmoid')(g)
generator = Model([g_input, g_input_cond], g_output)
g_opt = Adam(lr=1e-4)
generator.compile(loss='binary_crossentropy', optimizer=g_opt)

# discriminator
g_output_shape = X_train.shape[1:]
d_input = Input(shape=g_output_shape)
d_input_cond = Input(shape=(10, ))
d = Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2))(d_input)
d = PReLU()(d)
d = Dropout(0.2)(d)
d = Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2))(d)
d = PReLU()(d)
d = Dropout(0.2)(d)
d = Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2))(d)
d = PReLU()(d)
d = Flatten()(d)
d = Dropout(0.2)(d)
d = Dense(128)(merge([d, d_input_cond], mode='concat', concat_axis=1))
d = PReLU()(d)
d = Dropout(0.5)(d)
d_output = Dense(2, activation='softmax')(d)
discriminator = Model([d_input, d_input_cond], d_output)
d_opt = Adam(lr=1e-3)
discriminator.compile(loss='categorical_crossentropy', optimizer=d_opt)


# zamrzavanje/odmrzavanje slojeva
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# ceo GAN model
gan_input = Input(shape=(100, ))
gan_input_cond = Input(shape=(10, ))
gen_output = generator([gan_input, gan_input_cond])
gan_output = discriminator([gen_output, gan_input_cond])
gan = Model([gan_input, gan_input_cond], gan_output)
gan.compile(loss='categorical_crossentropy', optimizer=g_opt)


def plot_loss(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses['d'], label='discriminitive loss')
    plt.plot(losses['g'], label='generative loss')
    plt.legend()
    plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


# recnik za cuvanje vrednosti funkcije greske
losses = {'d': [], 'g': []}

plt_noise = np.random.uniform(0, 1, size=[100, 100])
plt_cond = np.tile(np.arange(0, 10), 10).flatten()
plt_cond = np_utils.to_categorical(plt_cond, 10)

output_imgs_dir = 'imgs/mnist_cnn_gan_cond'
if not os.path.exists(output_imgs_dir):
    os.mkdir(output_imgs_dir)


# glavna petlja za obucavanje
def train_for_n(nb_epoch=5000, start_at=0, batch_size=32):
    for e in tqdm(range(start_at, start_at+nb_epoch)):
        # generator izgenerise slike
        take_idx = np.random.randint(0, X_train.shape[0], size=batch_size)
        image_batch = X_train[take_idx, :, :, :]
        cond_batch = y_train[take_idx]
        cond_batch = np_utils.to_categorical(cond_batch, 10)
        noise_gen = np.random.uniform(0, 1, size=(batch_size, 100))
        generated_images = generator.predict([noise_gen, cond_batch])

        if e % 100 == 0:
            # plotovati rezultate generatora na svakih 100 epoha
            plt_generated_images = generator.predict([plt_noise, plt_cond])
            image = combine_images(plt_generated_images)
            image *= 255.0
            Image.fromarray(image.astype(np.uint8)).save(output_imgs_dir + '/epoch_{}.jpg'.format(e))

        # --- obucavanje diskriminatora ---
        # spajanje realnih i izgenerisanih primera
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * batch_size, 2])
        # prva polovina primera su realni, druga polovina su izgenerisani
        y[0:batch_size, 1] = 1
        y[batch_size:, 0] = 1

        # omoguciti obucavanje diskriminatora
        make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch([X, np.concatenate((cond_batch, cond_batch))], y)
        losses['d'].append(d_loss)

        # --- obucavanje generatora ---
        # generisanje uniformnog suma
        noise_tr = np.random.uniform(0, 1, size=[batch_size, 100])
        y2 = np.zeros([batch_size, 2])
        # zelimo da izlaz iz generatora bude klasifikovan kao da je iz realnog skupa podataka (varamo diskriminator)
        y2[:, 1] = 1

        # onemoguciti obucavanje diskriminatora
        make_trainable(discriminator, False)
        g_loss = gan.train_on_batch([noise_tr, cond_batch], y2)
        losses['g'].append(g_loss)


# obucavanje 10k epoha sa originalnim LR
train_for_n(nb_epoch=10000, start_at=0, batch_size=32)

# obucavanje 5k epoha sa smanjenim LR
K.set_value(g_opt.lr, 1e-5)
K.set_value(d_opt.lr, 1e-4)
train_for_n(nb_epoch=5000, start_at=10000, batch_size=32)

# obucavanje 5k epoha sa jos smanjenim LR
K.set_value(g_opt.lr, 1e-6)
K.set_value(d_opt.lr, 1e-5)
train_for_n(nb_epoch=5000, start_at=15000, batch_size=32)

# plotovanje kriva obucavanja
plot_loss(losses)
