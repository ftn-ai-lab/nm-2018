from __future__ import print_function

import numpy as np
import os
from skimage.io import imsave
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.callbacks import Callback

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 80

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# skaliranje na [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# definisanje autoenkodera
# prvo enkoder
inpt = Input((784, ))
encoder = Dense(128)(inpt)
encoder = Activation('relu')(encoder)
encoder = Dense(32)(encoder)
encoder = Activation('relu')(encoder)

# zatim dekoder
decoder = Dense(128)(encoder)
decoder = Activation('relu')(decoder)
decoder = Dense(784)(decoder)
decoder = Activation('sigmoid')(decoder)

# povezivanje enkodera i dekodera u autoenkoder
model = Model(input=inpt, output=decoder)

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# uzimamo nasumicnih 10 primera iz testnog skupa za vizualizaciju rezultata
idx = np.random.randint(X_test.shape[0], size=(10, ))
input_vect = X_test[idx]
input_imgs = input_vect.reshape(input_vect.shape[0], 28, 28)


output_imgs_dir = 'imgs/mnist_mlp_autoencoder'
if not os.path.exists(output_imgs_dir):
    os.mkdir(output_imgs_dir)


# utility funkcija za iscrtavanje rezultata: ulazna slika -> rekonstruisana slika
def generate_images(epoch):
    output_vect = model.predict(input_vect)
    output_imgs = output_vect.reshape(output_vect.shape[0], 28, 28)
    combined_img = combine_images(input_imgs, output_imgs)

    imsave(output_imgs_dir + '/epoch_{}.jpg'.format(epoch), combined_img)


def combine_images(inpt_imgs, outpt_imgs):
    num = inpt_imgs.shape[0]
    width = num
    height = 2
    shape = inpt_imgs.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=inpt_imgs.dtype)
    combined = np.concatenate((inpt_imgs, outpt_imgs))
    for index, img in enumerate(combined):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image


# Callback klasa iscrtavanje rezultata autoencodera
class CombineImagesCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        generate_images(epoch)

# obucavanje autoenkodera
model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, X_test), callbacks=[CombineImagesCallback()])
