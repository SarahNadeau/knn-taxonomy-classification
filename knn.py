# This script takes in standard vectorized train/test reads and performs KNN classification
    # can take in either vectorized genomes or 150 bp samples for training
    # always tests with 150 bp samples

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
from numpy import genfromtxt


# load vectorized sequences and enumerated class labels from standardized train/test set
def load_data():
    # to fetch standardized 150 bp train and test vectorized seqs:
    X_test = genfromtxt(
        '/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/vectorized_train_test/150bp_400seqs_6n_256bins_xTe.csv',
        delimiter=',')
    X_train = genfromtxt(
        '/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/vectorized_train_test/150bp_400seqs_6n_256bins_xTr.csv',
        delimiter=',')
    y_test = genfromtxt(
        '/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/vectorized_train_test/150bp_400seqs_6n_256bins_yTe.csv',
        delimiter=',')
    y_train = genfromtxt(
        '/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/vectorized_train_test/150bp_400seqs_6n_256bins_yTr.csv',
        delimiter=',')

    X_test = X_test.transpose()
    X_train = X_train.transpose()
    y_test = y_test.transpose()
    y_train = y_train.transpose()

    return X_test, X_train, y_test, y_train


def main():

    # set parameters
    n = None  # manually entered
    n_neighbors = 5
    # bins = None  # manually entered

    # generate samples
    X_test, X_train, y_test, y_train = load_data()
    print("X test shape: {}".format(X_test.shape))
    print("X train shape: {}".format(X_train.shape))
    print("y test shape: {}".format(y_test.shape))
    print("y train shape: {}".format(y_train.shape))

    # plot n-gram separation between organisms
    # pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    # transformed = pca.fit_transform(X_train)
    # y_colors = list(y_train)
    # plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
    # plt.title("{}-grams".format(n))
    # plt.show()

    # classify sequence by nearest neighbor in database
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    score = knn.fit(X_train, y_train).score(X_test, y_test)
    print('Testing error: {}'.format((1 - score)*100))
    print('with ** manually entered ** n-gram length = {}'.format(n))
    print('with K = {}'.format(n_neighbors))


if __name__ == '__main__':
    main()