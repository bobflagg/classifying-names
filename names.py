# -*- coding: utf-8 -*-

import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import string
import torch
from torch.utils.data import TensorDataset, DataLoader
import unicodedata

RANDOM_STATE = 4321

class NamesDataset(object):

    def __init__(self, directory='./data/names/', test_size=0.10, random_state=RANDOM_STATE, verbose=False):
        self.directory = directory
        self.vocabulary = string.ascii_letters + " .,;'"
        self.test_size = test_size
        self.random_state = random_state
        random.seed(random_state)
        self.verbose = verbose

        self.num_chars = len(self.vocabulary)
        self.char2int = {ch: ii for ii, ch in enumerate(self.vocabulary)}
        self.int2char = {ii: ch for ch, ii in self.char2int.items()}

        self.languages, language2names = self.load_data()
        X_train, y_train, X_test, y_test = self.build_train_test_split(self.languages, language2names)
        self.seq_len = np.max([len(name) for name in X_train + X_test])
        self.embed_names(X_train, X_test)
        self.encode_labels(y_train, y_test)

    def unicodeToAscii(self, s):
        """
        Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.vocabulary
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def load_data(self):
        if self.verbose: print("Building the language2names dictionary from data in %s ..." % self.directory)

        language2names = {}
        languages = []

        for fname in os.listdir(self.directory):
            path = os.path.join(self.directory, fname)
            language = os.path.splitext(os.path.basename(fname))[0]
            languages.append(language)
            lines = self.readLines(path)
            language2names[language] = lines

        if self.verbose:
            n_languages = len(languages)
            print("%3s %12s  %4s" % ("  ", "language", "cnt"))
            for i, language in enumerate(languages):
                print("\t%2d. %12s: %4d" % (i + 1, language, len(language2names[language])))

        return languages, language2names

    def build_train_test_split(self, languages, language2names):
        if self.verbose: print("Building train/test split ...")

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        if self.verbose: print(" %12s   #train    #test" % ("language",))
        for i, language in enumerate(languages):
            names_train, names_test = train_test_split(
                language2names[language],
                test_size=self.test_size,
                random_state=self.random_state
            )
            X_train.extend(names_train)
            y_train.extend([language] * len(names_train))

            X_test.extend(names_test)
            y_test.extend([language] * len(names_test))

            if self.verbose: print("\t%12s: # train: %5d, # test: %4d" % (language, len(names_train), len(names_test)))

        if self.verbose: print("\t%12s: # train: %5d, # test: %4d\n\n" % ("All", len(X_train), len(X_test)))

        Z = list(zip(X_train, y_train))
        random.shuffle(Z)
        X_train, y_train = zip(*Z)

        Z = list(zip(X_test, y_test))
        random.shuffle(Z)
        X_test, y_test = zip(*Z)

        return X_train, y_train, X_test, y_test

    def names2codes(self, names):
        names_encoded = []
        for name in names: names_encoded.append([self.char2int[c] for c in name])
        return names_encoded

    def codes2name(self, name):
        return "".join(self.int2char[i] for i in name)

    def codes2names(self, names):
        return [self.codes2name(name) for name in names]

    def encode_names(self, X_train, X_test):
        if self.verbose: print("Encoding names ...")
        X_train_encoded = self.names2codes(X_train)
        if self.verbose:
            print("\t# train names encoded:", len(X_train_encoded))
            n = 10
            X_train_decoded = self.codes2names(X_train_encoded[:n])
            for i in range(n): print("%19s" % (X_train_decoded[i], ))

        X_test_encoded = self.names2codes(X_test)
        if self.verbose:
            print("\t# test names encoded:", len(X_test_encoded))
            n = 10
            X_test_decoded = self.codes2names(X_test_encoded[:n])
            for i in range(n): print("%19s" % (X_test_decoded[i], ))

        return X_train_encoded, X_test_encoded

    def names2embeddings(self, names):
        names_encoded = self.names2codes(names)
        names_embedded = np.zeros((len(names), self.seq_len, self.num_chars), dtype=np.float32)
        index = 0
        for name in names_encoded:
            position = self.seq_len - len(name)
            for char in name:
                names_embedded[index, position, char] = 1
                position += 1
            index += 1
        return torch.from_numpy(names_embedded)

    def embedding2name(self, values, indices):
        return self.codes2name([int(indices[i]) for i in range(self.seq_len) if values[i] == 1])

    def embeddings2names(self, names):
        batch_size = names.shape[0]
        values, indices = torch.topk(names, 1)
        values = values.squeeze()
        indices = indices.squeeze()
        return [
            self.embedding2name(values[i], indices[i]) for i in range(batch_size)
        ]

    def embed_names(self, X_train, X_test):

        X_train_embedded = self.names2embeddings(X_train)
        if self.verbose:
            print("\tX_train_embedded:", X_train_embedded.shape, ";", X_train_embedded.dtype)
            n = 10
            some_names = self.embeddings2names(X_train_embedded[:n])
            for i in range(n):
                print("%19s" % (some_names[i], ))

        X_test_embedded = self.names2embeddings(X_test)
        if self.verbose:
            print("\n\tX_test_embedded:", X_test_embedded.shape, ";", X_test_embedded.dtype)
            n = 10
            some_names = self.embeddings2names(X_test_embedded[:n])
            for i in range(n):
                print("%19s" % (some_names[i], ))

        self.X_train_embedded, self.X_test_embedded = X_train_embedded, X_test_embedded

    def labels2encodings(self, labels):
        encoded_labels = np.array([self.class2code[label] for label in labels], dtype=np.float32)
        return torch.from_numpy(encoded_labels)

    def codes2labels(self, labels):
        return [self.code2class[int(i)] for i in labels]

    def encode_labels(self, y_train, y_test):
        classes = list(set(y_train + y_test))
        classes.sort()
        self.num_classes = len(classes)
        self.classes = classes
        self.class2code = {l:i for i, l in enumerate(classes)}
        self.code2class = {i:l for i, l in enumerate(classes)}

        self.y_train_encoded = self.labels2encodings(y_train)
        if self.verbose:
            print("encoded train labels:", self.y_train_encoded.shape, ";", self.y_train_encoded.dtype)
            n = 10
            labels = self.codes2labels(self.y_train_encoded[:n])
            for i in range(n): print("\t%s --> %d" % (labels[i], self.y_train_encoded[i]))


        self.y_test_encoded = self.labels2encodings(y_test)
        if self.verbose:
            print("encoded test labels:", self.y_test_encoded.shape, ";", self.y_test_encoded.dtype)
            n = 10
            labels = self.codes2labels(self.y_test_encoded[:n])
            for i in range(n): print("\t%s --> %d" % (labels[i], self.y_test_encoded[i]))

    def build_data_loaders(self, batch_size):
        trainset = TensorDataset(self.X_train_embedded, self.y_train_encoded)
        train_loader = DataLoader(trainset, shuffle=False, batch_size=batch_size)

        testset = TensorDataset(self.X_test_embedded, self.y_test_encoded)
        test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

        return train_loader, test_loader

def predict(dataset, model, name, rnn=True):
    inputs = dataset.names2embeddings([name])
    model.eval()
    if rnn: model.init_hidden()
    if torch.cuda.is_available(): inputs = inputs.cuda()
    output = model(inputs)
    values, indices = torch.topk(output, 1)
    language = dataset.codes2labels(indices.cpu().data.numpy())[0]
    return language

if __name__ == "__main__":
    batch_size = 16
    dataset = NamesDataset(verbose=False)
    train_loader, test_loader = dataset.build_data_loaders(batch_size)
    for i, data in enumerate(test_loader, 0):
        X, y = data
        break

    X, y = iter(train_loader).next()
    print("Train names and labels batch:", X.shape, "/", X.dtype, ";", y.shape, "/", y.dtype)
    names = dataset.embeddings2names(X)
    labels = dataset.codes2labels(y)
    for i in range(batch_size):
        print("\t%d. %19s --> %s" % (i, names[i], labels[i]))
    print('')
    X, y = iter(test_loader).next()
    print("Test names and labels batch:", X.shape, "/", X.dtype, ";", y.shape, "/", y.dtype)
    names = dataset.embeddings2names(X)
    labels = dataset.codes2labels(y)
    for i in range(batch_size):
        print("\t%d. %19s --> %s" % (i, names[i], labels[i]))
