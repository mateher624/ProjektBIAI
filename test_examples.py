from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Flatten, Dense
from keras import backend as K
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam

import os
import pickle
import matplotlib.pyplot as plt
import numpy.random as rng
import numpy as np

PATH = ""

def init_bias(shape, name=None):
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)

def init_weights(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)

# Budowanie sieci neuronowej: Siamese-Convnet

input_model = (105, 105, 1)                                                                                             # input tensora tablica 105x105 z jednym outputem
input_right = Input(input_model)                                                                                         # lewy input
input_left = Input(input_model)                                                                                        # prawy input
convnet = Sequential()                                                                                                  # początek modelu "nogi" sieci
convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_model, kernel_initializer=init_weights, kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=init_weights, bias_initializer=init_bias))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=init_weights, kernel_regularizer=l2(2e-4), bias_initializer=init_bias))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=init_weights, kernel_regularizer=l2(2e-4), bias_initializer=init_bias))
convnet.add(Flatten())
convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=init_weights, bias_initializer=init_bias))

merge_func = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
leg_right = convnet(input_right)                                                                                         # dodanie lewej nogi
leg_left = convnet(input_left)                                                                                        # dodanie prawej nogi
merge_layer = merge_func([leg_right, leg_left])                                                                          # warstwa łączenia outputów
prediction = Dense(1, activation='sigmoid', bias_initializer=init_bias)(merge_layer)                                       # połączenie outputów sieci za pomocą funkcji
siamese_net = Model(inputs=[input_right, input_left], outputs=prediction)                                               # model

optimizer = Adam(0.00006)                                                                                               # algorytm optymailizacji: Adam (learning rate)

siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)                                                    # kompilacja

siamese_net.count_params()                                                                                              # log

                                                                                                              # scieżka do pikli

#with open(os.path.join(PATH, "train.pickle"), "rb") as file:
#    (data, dictionary) = pickle.load(file)

#with open(os.path.join(PATH, "val.pickle"), "rb") as file:
#    (val_data, val_dictionary) = pickle.load(file)

def concat_images(X):
    nc, h, w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n * w, n * h))
    x = 0
    y = 0
    for example in range(nc):
        img[x * w:(x + 1) * w, y * h:(y + 1) * h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img

def plot_oneshot_task(pairs):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(105, 105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

class Siamese_Loader:

    def __init__(self, path, data_subsets=["train", "val"]):
        self.data = {}
        self.letter_dictionary = {}
        self.info = {}

        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path, "rb") as f:
                (X, c) = pickle.load(f)
                self.data[name] = X
                self.letter_dictionary[name] = c

    def fancy_test_good(self, model, language=None, s ="val"):
        dataset = self.data[s]
        n_letters, n_examples, w, h = dataset.shape
        if language is not None:
            range_start, range_end = self.letter_dictionary[s][language]
            letters = rng.choice(range(range_start, range_end), size=(1,), replace=False)
        else:
            letters = rng.choice(range(n_letters), size=(1,), replace=False)
        letter = letters[0]  # wybrana literka
        position1, position2 = rng.choice(n_examples, replace=False, size=(2,))
        first_image = np.asarray([dataset[letter, position1, :, :]] * 1).reshape(1, w, h, 1)
        second_image = np.asarray([dataset[letter, position2, :, :]] * 1).reshape(1, w, h, 1)
        pairs = [first_image, second_image]
        plot_oneshot_task(pairs)
        probs = model.predict(pairs)
        result = int(round(probs[0][0]))
        print("Result of comparing: ", result)

    def fancy_test_bad(self, model, language=None, s ="val"):
        X = self.data[s]
        n_letters, n_examples, w, h = X.shape
        if language is not None:
            range_start, range_end = self.letter_dictionary[s][language]
            letters = rng.choice(range(range_start, range_end), size=(2,), replace=False)
        else:
            letters = rng.choice(range(n_letters), size=(2,), replace=False)
        letter1 = letters[0]  # wybrana literka1
        letter2 = letters[1]  # wybrana literka2
        position1, position2 = rng.choice(n_examples, replace=False, size=(2,))
        first_image = np.asarray([X[letter1, position1, :, :]] * 1).reshape(1, w, h, 1)
        second_image = np.asarray([X[letter2, position2, :, :]] * 1).reshape(1, w, h, 1)
        pairs = [first_image, second_image]
        plot_oneshot_task(pairs)
        probs = model.predict(pairs)
        result = int(round(probs[0][0]))
        print("Result of comparing: ", result)

loader = Siamese_Loader(PATH)
weights_path = os.path.join(PATH, "weights")
siamese_net.load_weights(weights_path)

# test example
loader.fancy_test_good(siamese_net, "Ztestimg", "train")
