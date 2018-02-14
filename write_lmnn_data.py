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
    n_gram_len = 5
    n_neighbors = 4
    num_test_seqs = 100
    seq_len = 150
    # bins = 5**n_gram_len
    bins = 3125

    # generate samples
    X, y = GetTrainData.get_seq(seq_len, n_gram_len, num_test_seqs, bins)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
    print("{} training sequences".format(len(X_train)))
    print("{} testing sequences".format(len(X_test)))

    # write files for LMNN input
    X_tr = np.asarray(X_train)
    X_tr = np.transpose(X_tr)
    np.savetxt("/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/xTr.csv",
               X_tr, delimiter=",")

    X_te = np.asarray(X_test)
    X_te = np.transpose(X_te)
    np.savetxt("/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/xTe.csv",
               X_te, delimiter=",")

    # need to make labels numeric
    f = open("/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/yTr.csv", 'w')
    for i in range(0, len(y_train)):
        numeric = []
        for char in y_train[i]:
            numeric.append(str(ord(char)))
        numeric = "".join(numeric)
        if i < len(y_train) - 1:
            f.write(numeric + ',')
        else:
            f.write(numeric)
    f.close

    f = open("/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/yTe.csv", 'w')
    for i in range(0, len(y_test)):
        numeric = []
        for char in y_test[i]:
            numeric.append(str(ord(char)))
        numeric = "".join(numeric)
        if i < len(y_test) - 1:
            f.write(numeric + ',')
        else:
            f.write(numeric)
    f.close


if __name__ == '__main__':
    main()