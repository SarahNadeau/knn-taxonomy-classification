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
        # print(ngrams)
        return ngrams

class GetTrainData:
    def __init__(self):
        pass

    @staticmethod
    def get_train_fasta(n):
        for filename in os.listdir('Genomes'):
            if filename.endswith(".fna"): ###
                filepath = os.path.join('Genomes', filename)
                seqs = []
                num_seqs = 0
                for seq_record in SeqIO.parse(filepath, 'fasta'):
                    id = seq_record.id
                    print(id)
                    seq = seq_record.seq
                    seqs.append(seq)

                X = []
                y = []
                scaffold_vecs = []
                scaffold_lengths = []
                for seq in seqs:
                    scaffold_vecs.append(np.array(list(VectorizeMethods.make_ngram(seq, n).values())))
                    scaffold_lengths.append((len(seq) - n + 1))
                vec = sum(scaffold_vecs)/sum(scaffold_lengths)
                X.append(vec)
                y.append(id)

        return X, y

class GenerateSeq:
    def __init__(self):
        pass

    # This function takes in a filepath and returns a DNA sequence of length k, genome ID
    @staticmethod
    def get_genbank(k, filepath, filetype):
        for seq_record in SeqIO.parse(filepath, filetype):
            # note - only returns sequence from first seq_record if multiple
            rand_start = np.random.randint(0, len(seq_record.seq) - k + 1)
            return seq_record.seq[rand_start:rand_start + k], seq_record.id

    # This function takes in a HiSeq filepath and returns all fasta DNA sequences, fasta IDs
    @staticmethod
    def get_fasta(filepath, num_test_seqs):
        ids = []
        seqs = []
        num_seqs = 0
        for seq_record in SeqIO.parse(filepath, 'fasta'):
            if num_seqs < num_test_seqs:
                ids.append(seq_record.id)
                print(seq_record.id)
                # note - only returns sequence from first seq_record if multiple
                seq = seq_record.seq
                seqs.append(seq)
                num_seqs += 1
            else:
                break
        return seqs, ids

    @staticmethod
    def get_fastq(filepath, num_test_seqs):
        ids = []
        seqs = []
        num_seqs = 0
        for seq_record in SeqIO.parse(filepath, 'fastq'):
            if num_seqs < num_test_seqs:
                ids.append(seq_record.id)
                # note - only returns sequence from first seq_record if multiple
                seq = seq_record.seq
                seqs.append(seq)
                num_seqs += 1
            else:
                break
        return seqs, ids

class GetXSeqsYIDs:
    def __init__(self):
        pass

    # This function returns array of vectorized DNA sequences (X), array of sequence IDs (y)
    # for genbank files in directory
    @staticmethod
    def get_genbank_arrays(num_test_seqs, len_test_seqs, n):
        org_num = 0
        X = []
        y = []
        for filename in os.listdir('Genomes'):
            if filename.endswith('.gbff'):
                filename = os.path.join('Genomes', filename)
                for i in range(0, num_test_seqs):
                    test_seq, test_id = GenerateSeq.get_genbank(len_test_seqs, filename, 'genbank')
                    test_vec = np.array(list(VectorizeMethods.make_ngram(test_seq, n).values())) / (len(test_seq) - n + 1)
                    X.append(test_vec)
                    y.append(test_id)
                org_num += 1

        return X, y

    @staticmethod
    # This function returns array of vectorized DNA sequences (X), array of sequence IDs (y)
    # for file HiSeq_accuracy.fa in directory
    def get_fasta_arrays(num_test_seqs, n):
        X = []
        y = []
        for filename in os.listdir('Genomes'):
            if filename.endswith(".fna"): ###
                print(filename)
                filename = os.path.join('Genomes', filename)
                seq_list, id_list = GenerateSeq.get_fasta(filename, num_test_seqs)
                for seq in seq_list:
                    vec = np.array(list(VectorizeMethods.make_ngram(seq, n).values())) / (len(seq) - n + 1)
                    X.append(vec)
                for id in id_list:
                    id = id.split('.')[0]
                    y.append(id)
        return X, y

    @staticmethod
    # This function returns array of vectorized DNA sequences (X), array of sequence IDs (y)
    # for file HiSeq_accuracy.fa in directory
    def get_fastq_arrays(num_test_seqs, n):
        X = []
        y = []
        for filename in os.listdir('Genomes'):
            if filename.endswith(".fq"):
                filename = os.path.join('Genomes', filename)
                seq_list, id_list = GenerateSeq.get_fastq(filename, num_test_seqs)
                for seq in seq_list:
                    vec = np.array(list(VectorizeMethods.make_ngram(seq, n).values())) / (len(seq) - n + 1)
                    X.append(vec)
                for id in id_list:
                    id = id.split('.')[0]
                    y.append(id)
        print(y)
        return X, y


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
        len_test_seqs = 150 # applies only to sequences fetched from genbank files
        # test filetype options are 'genbank', 'fasta', 'fastq'
        test_filetype = 'fasta'
        print(" {}-grams, \n number of neighbors = {}, \n number of test sequences = {}, \n test filetype = {} \n"
              .format(n, n_neighbors, num_test_seqs, test_filetype))
    else:
        n = input("Enter n for n-gram vectorization of reads: ")
        n_neighbors = input("Enter number of neighbors for KNN: ")
        num_test_seqs = input("Enter number of test sequences to generate: ")
        test_filetype = input("Enter a valid filetype ('genbank', 'fasta', or 'fastq': ")

    GetTrainData.get_train_fasta(n)

    # generate arrays (X, y) of vectorized sequences and genome IDs
    if test_filetype == "genbank":
        X, y = GetXSeqsYIDs.get_genbank_arrays(num_test_seqs, len_test_seqs, n)
    elif test_filetype == 'fasta':
        X, y = GetXSeqsYIDs.get_fasta_arrays(num_test_seqs, n)
        print (X) ###
        print(y) ###
        print("number of sequences vectorized: {}".format(len(X)))
    elif test_filetype == 'fastq':
        X, y = GetXSeqsYIDs.get_fastq_arrays(num_test_seqs, n)
        print("number of sequences vectorized: {}".format(len(X)))

    # plot n-gram separation between organisms
    # pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    # transformed = pca.fit_transform(X)
    # y_colors = list(map(lambda x: sum_chars(x), y))
    # plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
    # plt.title("< insert file name >, {}-grams, {} neighbors".format(n, n_neighbors))
    # plt.show()

    # split X an y into training and test set for KNN
    # eventually training set should be NCBI reference genomes and testing set should be metagenomic samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

    # # classify sequence by nearest neighbor in database
    # knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    # print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
    # print('with n = {}'.format(n))
    # print('with n_neighbors = {}'.format(n_neighbors))


if __name__ == '__main__':
    main()