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


class GetTrainData:
    def __init__(self):
        pass

    @staticmethod
    def get_seq(k, n, num_seqs, bins):
        X = []
        y = []
        for filename in os.listdir('Genomes'):
            if filename.endswith(".fasta.txt"):
                filepath = os.path.join('Genomes', filename)

                for seq_record in SeqIO.parse(filepath, 'fasta'):
                    for i in range(0, num_seqs):
                        # note - only returns sequence from last seq_record if multiple
                        rand_start = np.random.randint(0, len(seq_record.seq) - k + 1)
                        seq = seq_record.seq[rand_start:rand_start + k]
                        vec_seq = VectorizeMethods.make_ngram(seq, n, bins)
                        X.append(vec_seq)
                        id = seq_record.id.split("|")[1].split("|")[0]
                        y.append(id)
        return X, y


# This function is used to convert tax labels to ints for plotting legend
def sum_chars(word):
    num = 0
    for char in word:
        num += ord(char)
    return num


def main():

    # set parameters
    n_gram_len = 10
    n_neighbors = 4
    num_test_seqs = 5
    seq_len = 150
    # bins = 5**n_gram_len
    bins = 5000

    # generate samples
    X, y = GetTrainData.get_seq(seq_len, n_gram_len, num_test_seqs, bins)

    # reduce dimensions by hashing


    # plot n-gram separation between organisms
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pca.fit_transform(X)
    y_colors = list(map(lambda x: sum_chars(x), y))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
    plt.title("{}-grams with {} bins".format(n_gram_len, bins))
    plt.show()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

    # with open('X_test5.pickle', 'wb') as f:
    #     pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
    #
    # with open('X_train5.pickle', 'wb') as f:
    #     pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
    #
    # with open('y_test5.pickle', 'wb') as f:
    #     pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)
    #
    # with open('y_train5.pickle', 'wb') as f:
    #     pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)

    # with open('X_test5.pickle', 'rb') as f:
    #     X_test = pickle.load(f)
    #
    # with open('X_train5.pickle', 'rb') as f:
    #     X_train = pickle.load(f)
    #
    # with open('y_test5.pickle', 'rb') as f:
    #     y_test = pickle.load(f)
    #
    # with open('y_train5.pickle', 'rb') as f:
    #     y_train = pickle.load(f)

    # print(len(X_test[0]))

    # falconn.get_default_parameters()

    # classify sequence by nearest neighbor in database
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    score = knn.fit(X_train, y_train).score(X_test, y_test)
    print('KNN score: %f' % score)
    print('with n_gram length = {}'.format(n_gram_len))
    print('with number of bins = {}'.format(bins))
    print('with n_neighbors = {}'.format(n_neighbors))


if __name__ == '__main__':
    main()

    # try re-downloading fastq files and not opening with text editor (currently a \ at end of each line)