from Bio import SeqIO
import numpy as np
import os
import itertools
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import model_selection
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
import pickle
import falconn


class VectorizeMethods:
    def __init__(self):
        pass

    # This function takes in a DNA sequence and returns it as a 1 x 5**n vector
    @staticmethod
    def ngram_to_num(str, bins):
        num = 0
        for i in str:
            num = num * 5
            if i == 'A':
                num += 0
            elif i == 'T':
                num += 1
            elif i == 'C':
                num += 2
            elif i == 'G':
                num += 3
            elif i == 'N':
                num += 4
        return hash(num)%bins

    @staticmethod
    def make_ngram(seq, n, bins):
        vec = np.zeros(bins)
        for s in range(0, len(seq) - n + 1):
            ngram = seq[s:s + n]
            vec[VectorizeMethods.ngram_to_num(ngram, bins=bins)] += 1
        vec = np.divide(vec, len(seq) - n + 1)
        return vec


# translate string data labels to ints
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(set(sorted(y_str)))])
    return [ylabel_dict[x] for x in y_str]


# load data from standardized train/test set
def load_data():
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_test150.pickle', 'rb') as f:
        x_test = pickle.load(f)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_train150.pickle', 'rb') as f:
        x_train = pickle.load(f)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_test150.pickle', 'rb') as f:
        y_test_str = pickle.load(f)
        y_test = enumerate_y_labels(y_test_str)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_train150.pickle', 'rb') as f:
        y_train_str = pickle.load(f)
        y_train = enumerate_y_labels(y_train_str)
        f.close()
    return x_test, x_train, y_test, y_train


def main():

    # set parameters
    n = 3
    n_neighbors = 20
    bins = 5**n  # bins = 5**n to have no reduction in dimensionality

    # generate samples
    # X, y = GetTrainData.get_seq(seq_len, n_gram_len, num_test_seqs, bins)
    x_test, x_train, y_test, y_train = load_data()

    # vectorize sequences
    X_train = []
    for seq in x_train:
        X_train.append(VectorizeMethods.make_ngram(seq, n, bins=bins))
    X_test = []
    for seq in x_test:
        X_test.append(VectorizeMethods.make_ngram(seq, n, bins=bins))

    # plot n-gram separation between organisms
    # pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    # transformed = pca.fit_transform(X_train)
    # y_colors = list(y_train)
    # plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
    # plt.title("{}-grams with {} bins".format(n, bins))
    # plt.show()

    # classify sequence by nearest neighbor in database
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    score = knn.fit(X_train, y_train).score(X_test, y_test)
    print('Testing error: {}'.format(1 - score))
    print('with n-gram length = {}'.format(n))
    print('with number of bins = {}'.format(bins))
    print('with n_neighbors = {}'.format(n_neighbors))


if __name__ == '__main__':
    main()