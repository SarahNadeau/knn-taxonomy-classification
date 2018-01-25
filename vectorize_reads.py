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


class GenerateSeq:
    def __init__(self):
        pass

    # This function takes in a filepath and returns a DNA sequence of length k, genome ID
    @staticmethod
    def get_seq(k, filepath, filetype):
        for seq_record in SeqIO.parse(filepath, filetype):
            # note - only returns sequence from first seq_record if multiple
            rand_start = np.random.randint(0, len(seq_record.seq) - k + 1)
            return seq_record.seq[rand_start:rand_start + k], seq_record.id

    # This function takes in a HiSeq filepath and returns all fasta DNA sequences, fasta IDs
    @staticmethod
    def get_HiSeq(filepath):
        ids = []
        seqs = []
        for seq_record in SeqIO.parse(filepath, 'fasta'):
            ids.append(seq_record.id)
            # note - only returns sequence from first seq_record if multiple
            seq = seq_record.seq
            seqs.append(seq)
        return seqs, ids

class GetXSeqsYIDs:
    def __init__(self):
        pass

    # This function returns array of vectorized DNA sequences (X), array of sequence IDs (y)
    # for genbank files in directory
    @staticmethod
    def get_seq_arrays(num_test_seqs, len_test_seqs, n):
        org_num = 0
        X = []
        y = []
        for filename in os.listdir('Genomes'):
            if filename.endswith('.gbff'):
                filename = os.path.join('Genomes', filename)
                for i in range(0, num_test_seqs):
                    test_seq, test_id = GenerateSeq.get_seq(len_test_seqs, filename, 'genbank')
                    test_vec = np.array(list(VectorizeMethods.make_ngram(test_seq, n).values())) / (len(test_seq) - n + 1)
                    X.append(test_vec)
                    y.append(test_id)
                org_num += 1

        return X, y

    @staticmethod
    # This function returns array of vectorized DNA sequences (X), array of sequence IDs (y)
    # for file HiSeq_accuracy.fa in directory
    def get_HiSeq_arrays(n):
        X = []
        y = []
        for filename in os.listdir('Genomes'):
            if filename == 'HiSeq_accuracy.fa':
                filename = os.path.join('Genomes', filename)
                Hi_seq_list, Hi_id_list = GenerateSeq.get_HiSeq(filename)
                for seq in Hi_seq_list:
                    Hi_vec = np.array(list(VectorizeMethods.make_ngram(seq, n).values())) / (len(seq) - n + 1)
                    X.append(Hi_vec)
                for id in Hi_id_list:
                    Hi_id = id.split('.')[0]
                    y.append(Hi_id)

        return X, y


# This function is used to convert tax labels to ints for plotting legend
def sum_chars(word):
    num = 0
    for char in word:
        num += ord(char)
    return num


def main():

    # set parameters
    n = 10
    n_neighbors = 1
    num_test_seqs = 10
    len_test_seqs = 150
    # test filetype options are 'genbank', 'HiSeq'
    test_filetype = 'HiSeq'

    # generate arrays (X, y) of vectorized sequences and genome IDs
    if test_filetype == "genbank":
        X, y = GetXSeqsYIDs.get_seq_arrays(num_test_seqs, len_test_seqs, n)
    elif test_filetype == 'HiSeq':
        X, y = GetXSeqsYIDs.get_HiSeq_arrays(n)

    # plot n-gram separation between organisms
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pca.fit_transform(X)
    y_colors = list(map(lambda x: sum_chars(x), y))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
    plt.title("{}-gram separation between 10 Hi-Seq genomes".format(n))
    plt.show()

    # split X an y into training and test set for KNN
    # eventually training set should be NCBI reference genomes and testing set should be metagenomic samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

    # classify sequence by nearest neighbor in database
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
    print('with n = {}'.format(n))
    print('with n_neighbors = {}'.format(n_neighbors))


if __name__ == '__main__':
    main()