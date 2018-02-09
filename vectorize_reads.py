from Bio import SeqIO
import numpy as np
import os
import itertools
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import model_selection
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA


class VectorizeMethods:
    def __init__(self):
        pass

    # This function takes in a DNA sequence and returns it as a 1 x 5**n vector
    @staticmethod
    def make_ngram(seq, n):
        bases = ['A', 'C', 'T', 'G', 'N']
        ngrams = {}
        for gram in itertools.product(bases, repeat=n):
            ngrams[''.join(gram)] = 0
        for s in range(0, len(seq) - n + 1):
            ngrams[seq[s:s + n]] += 1
        return ngrams


class GetTrainData:
    def __init__(self):
        pass

    @staticmethod
    def get_train_fasta(n):
        X_train = []
        y_train = []

        for filename in os.listdir('Genomes'):
            if filename.endswith(".fasta.txt"): ###
                filepath = os.path.join('Genomes', filename)

                scaffold_vecs = []
                scaffold_lengths = []

                for seq_record in SeqIO.parse(filepath, 'fasta'):
                    id = seq_record.id # ID is currently a scaffold id! Should be for whole genome!
                    y = id.split("|")[1].split("|")[0]
                    print(y)
                    seq = seq_record.seq
                    scaffold_vecs.append(np.array(list(VectorizeMethods.make_ngram(seq, n).values())))
                    scaffold_lengths.append((len(seq) - n + 1))

                genome_vec = sum(scaffold_vecs)/sum(scaffold_lengths)
                X_train.append(genome_vec)
                y_train.append(y)

        return X_train, y_train


class GetTestData:
    def __init__(self):
        pass

    # This function takes in a fastq file's filepath and returns all DNA sequences, IDs
    # designed to take in input from Metabenchmark study's simulated metagenomic reads
    @staticmethod
    def get_test_fastq(num_test_seqs, n):
        X_test = []
        y_test = []
        num_seqs = 0

        for filename in os.listdir('TestData'):
            if filename.endswith(".fq") or filename.endswith(".fastq"):
                filepath = os.path.join('TestData', filename)
                for seq_record in SeqIO.parse(filepath, 'fastq'):
                    if num_seqs < num_test_seqs:
                        id = seq_record.id.split('-')[0]
                        y_test.append(id)
                        seq = seq_record.seq
                        vec = np.array(list(VectorizeMethods.make_ngram(seq, n).values())) / (len(seq) - n + 1)
                        X_test.append(vec)
                        num_seqs += 1

        return X_test, y_test

    # This function fetches only specific test sequences with original or relative in training set
    # designed for Metabenchmark file setA1_1-0.fq.gz
    @staticmethod
    def get_test_fastq_specific(num_test_seqs, n):
        X_test = []
        y_test = []

        for filename in os.listdir('TestData'):
            if filename.endswith(".fq") or filename.endswith(".fastq"):
                filepath = os.path.join('TestData', filename)
                for seq_record in SeqIO.parse(filepath, 'fastq'):
                    id = seq_record.id.split('-')[0]
                    if id in ["AE016823", "AP006618", "AP010889", "CP000360", "CP004405", "FP929050"]:
                        y_test.append(id)
                        seq = seq_record.seq
                        vec = np.array(list(VectorizeMethods.make_ngram(seq, n).values())) / (len(seq) - n + 1)
                        X_test.append(vec)
        return X_test, y_test


# This function is used to convert tax labels to ints for plotting legend
def sum_chars(word):
    num = 0
    for char in word:
        num += ord(char)
    return num


def main():

    # set parameters
    use_defaults = True ###
    # use_defaults = input("Use default parameters? Enter True of False. \n")
    if use_defaults:
        n = 1
        n_neighbors = 1
        num_test_seqs = 20
        test_filetype = 'fastq'  # file type options are 'fasta', 'fastq' in folder 'TestData'
        # training data is assumed to be fasta.txt files in 'Genomes' folder
        print(" {}-grams, \n number of neighbors = {}, \n number of test sequences = {}, \n test filetype = {} \n"
              .format(n, n_neighbors, num_test_seqs, test_filetype))
    else:
        n = input("Enter n for n-gram vectorization of reads: ")
        n_neighbors = input("Enter number of neighbors for KNN: ")
        num_test_seqs = input("Enter number of test sequences to generate: ")
        test_filetype = input("Enter a valid filetype ('genbank', 'fasta', or 'fastq': ")


    # generate training data arrays (X_train, y_train) of vectorized genomes in file Genomes
    ### y_train needs to be same lables as y_test !!
    X_train, y_train = GetTrainData.get_train_fasta(n)

    # generate testing data arrays (X_test, y_test) of vectorized sequences and genome IDs
    X_test, y_test = GetTestData.get_test_fastq(num_test_seqs, n)

    print("number of test sequences: {}".format(len(X_test)))
    print("X for test sequences: {}".format(X_test))
    print("X for train sequences: {}".format(X_train))
    print("y for test sequences: {}".format(y_test))
    print("y for train sequences: {}".format(y_train))

    # plot n-gram separation between organisms
    # pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    # transformed = pca.fit_transform(X)
    # y_colors = list(map(lambda x: sum_chars(x), y))
    # plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
    # plt.title("< insert file name >, {}-grams, {} neighbors".format(n, n_neighbors))
    # plt.show()

    # classify sequence by nearest neighbor in database
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
    print('with n = {}'.format(n))
    print('with n_neighbors = {}'.format(n_neighbors))


if __name__ == '__main__':
    main()

    # try re-downloading fastq files and not opening with text editor (currently a \ at end of each line)