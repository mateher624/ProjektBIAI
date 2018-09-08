from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Flatten, Dense
from keras import backend
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.utils import shuffle

import os
import pickle
import matplotlib.pyplot as plt
import numpy.random as rng
import numpy as np

# path to stored data
PATH = ""


def init_bias(shape, name=None):
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return backend.variable(values, name=name)


def init_weights(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return backend.variable(values, name=name)


# Budowanie sieci neuronowej: Siamese-Convnet

input_model = (105, 105, 1)
input_right = Input(input_model)
input_left = Input(input_model)
convnet = Sequential()
convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_model, kernel_initializer=init_weights,
                   kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=init_weights,
                   bias_initializer=init_bias))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=init_weights, kernel_regularizer=l2(2e-4),
                   bias_initializer=init_bias))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=init_weights, kernel_regularizer=l2(2e-4),
                   bias_initializer=init_bias))
convnet.add(Flatten())
convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=init_weights,
                  bias_initializer=init_bias))

merge_func = Lambda(lambda tensors: backend.abs(tensors[0] - tensors[1]))
leg_right = convnet(input_right)
leg_left = convnet(input_left)
merge_layer = merge_func([leg_right, leg_left])
prediction = Dense(1, activation='sigmoid', bias_initializer=init_bias)(merge_layer)
siamese_net = Model(inputs=[input_right, input_left], outputs=prediction)

optimizer = Adam(0.00006)
siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

siamese_net.count_params()

with open(os.path.join(PATH, "train.pickle"), "rb") as file:
    (train_data, train_dictionary) = pickle.load(file)

with open(os.path.join(PATH, "val.pickle"), "rb") as file:
    (val_data, val_dictionary) = pickle.load(file)

print("Training set alphabet:")
print(train_dictionary.keys())
print("Validation set alphabet:")
print(val_dictionary.keys())


class SiameseLoader:

    def __init__(self, path, data_subsets=["train", "val"]):
        self.data = {}
        self.letter_dictionary = {}
        self.info = {}

        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("Loading data from {}".format(file_path))
            with open(file_path, "rb") as f:
                dataset, dictionary = pickle.load(f)
                self.data[name] = dataset
                self.letter_dictionary[name] = dictionary

    def get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""
        dataset = self.data[s]
        n_letters, n_examples, width, height = dataset.shape

        # randomly sample several classes to use in the batch
        letters = rng.choice(n_letters, size=(batch_size,), replace=False)
        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, height, width, 1)) for i in range(2)]
        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            letter = letters[i]
            position1 = rng.randint(0, n_examples)
            pairs[0][i, :, :, :] = dataset[letter, position1].reshape(width, height, 1)
            position2 = rng.randint(0, n_examples)
            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                chosen_letter = letter
            else:
                chosen_letter = (letter + rng.randint(1, n_letters)) % n_letters
            pairs[1][i, :, :, :] = dataset[chosen_letter, position2].reshape(width, height, 1)
        return pairs, targets

    #    def generate(self, batch_size, s="train"):
    #       """a generator for batches, so model.fit_generator can be used. """
    #       while True:
    #            pairs, targets = self.get_batch(batch_size, s)
    #           yield (pairs, targets)

    def make_oneshot_task(self, N, s="val", language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        dataset = self.data[s]
        n_letters, n_examples, width, height = dataset.shape
        indices = rng.randint(0, n_examples, size=(N,))
        if language is not None:
            range_start, range_end = self.letter_dictionary[s][language]
            if N > range_end - range_start:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            letters = rng.choice(range(range_start, range_end), size=(N,), replace=False)

        else:
            letters = rng.choice(range(n_letters), size=(N,), replace=False)
        letter = letters[0]
        position1, position2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([dataset[letter, position1, :, :]] * N).reshape(N, width, height, 1)
        support_set = dataset[letters, indices, :, :]
        support_set[0, :, :] = dataset[letter, position2]
        support_set = support_set.reshape(N, width, height, 1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets

    def test_oneshot(self, model, N, k, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k, N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))
        return percent_correct

# Instantiate the class
loader = SiameseLoader(PATH)


# In[4]: TO REMOVE
#def concat_images(X):
#    """Concatenates a bunch of images into a big matrix for plotting purposes."""
#    nc, h, w, _ = X.shape
#    X = X.reshape(nc, h, w)
#    n = np.ceil(np.sqrt(nc)).astype("int8")
#    img = np.zeros((n * w, n * h))
#    x = 0
#    y = 0
#    for example in range(nc):
#        img[x * w:(x + 1) * w, y * h:(y + 1) * h] = X[example]
#       y += 1
#        if y >= n:
#            y = 0
#            x += 1
#    return img


#def plot_oneshot_task(pairs):
#    """Takes a one-shot task given to a siamese net and  """
#    fig, (ax1, ax2) = plt.subplots(2)
#    ax1.matshow(pairs[0][0].reshape(105, 105), cmap='gray')
#    img = concat_images(pairs[1])
#    ax1.get_yaxis().set_visible(False)
#    ax1.get_xaxis().set_visible(False)
#    ax2.matshow(img, cmap='gray')
#    plt.xticks([])
#    plt.yticks([])
#    plt.show()


# example of a one-shot learning task
# pairs, targets = loader.make_oneshot_task(20, "train", "Futurama")
# plot_oneshot_task(pairs)

# In[5]:


# Training loop
print("Training started")

evaluate_every = 100  # interval for evaluating on one-shot tasks
loss_every = 1  # interval for printing loss (iterations)
batch_size = 32
n_iter = 1  # 90000
N_way = 20  # how many classes for testing one-shot tasks>
n_val = 250  # how many one-shot tasks to validate on?
best = -1
weights_path = os.path.join(PATH, "weights")

# not tested
print("Loading weights...")
siamese_net.load_weights(weights_path)

print("Training...")
for i in range(1, n_iter):
    (inputs, targets) = loader.get_batch(batch_size)
    loss = siamese_net.train_on_batch(inputs, targets)
    print(loss)
    if i % evaluate_every == 0:
        print("Evaluating...")
        val_acc = loader.test_oneshot(siamese_net, N_way, n_val, verbose=True)
        if val_acc >= best:
            print("Saving weights")
            siamese_net.save(weights_path)
            best = val_acc

    if i % loss_every == 0:
        print("Iteration: {}, training loss: {:.2f},".format(i, loss))  # log
print("Training completed.")

def nearest_neighbour_correct(pairs, targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    distances = np.zeros_like(targets)
    for i in range(len(targets)):
        distances[i] = np.sum(np.sqrt(pairs[0][i] ** 2 - pairs[1][i] ** 2))
    if np.argmin(distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways, n_trials, loader):
    """Returns accuracy of one shot """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials, N_ways))

    n_right = 0

    for i in range(n_trials):
        pairs, targets = loader.make_oneshot_task(N_ways, "val")
        correct = nearest_neighbour_correct(pairs, targets)
        n_right += correct
    return 100.0 * n_right / n_trials


ways = np.arange(1, 60, 2)
resume = False
val_accs, train_accs, nn_accs = [], [], []
trials = 450  # 450
for N in ways:
    val_accs.append(loader.test_oneshot(siamese_net, N, trials, "val", verbose=True))
    train_accs.append(loader.test_oneshot(siamese_net, N, trials, "train", verbose=True))
    nn_accs.append(test_nn_accuracy(N, trials, loader))

# plot the accuracy vs num categories for each
plt.plot(ways, val_accs, "m")
plt.plot(ways, train_accs, "y")
plt.plot(ways, nn_accs, "c")

plt.plot(ways, 100.0 / ways, "r")
# plt.show()

fig, ax = plt.subplots(1)
ax.plot(ways, val_accs, "m", label="Siamese(val set)")
ax.plot(ways, train_accs, "y", label="Siamese(train set)")
plt.plot(ways, nn_accs, label="Nearest neighbour")

ax.plot(ways, 100.0 / ways, "g", label="Random guessing")
plt.xlabel("Number of possible classes in one-shot tasks")
plt.ylabel("% Accuracy")
plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
inputs, targets = loader.make_oneshot_task(20, "val")
plt.show()

# print(inputs[0].shape)
# p = siamese_net.predict(inputs)
# print(p)

a = test_nn_accuracy(3, 500, loader)
print(a)
