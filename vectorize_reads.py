from Bio import SeqIO
import numpy as np
import os
import itertools
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import model_selection


class VectorizeMethods:
    def __init__(self):
        pass

    # This function takes in a DNA sequence and returns it as a 1 x 5**n vector
    @staticmethod
    def make_ngram(seq, n):
        bases = ['A', 'C', 'T', 'G', 'N']
        ngram = np.zeros(5**n)
        for s in range(0, len(seq) - n + 1):
            for i, gram in zip(range(0, 5**n), itertools.product(bases, repeat=n)):
                if seq[s:s + n] == ''.join(gram):
                    ngram[i] += 1
        return ngram


class GenerateSeq:
    def __init__(self):
        pass

    # This function takes in a filepath and generates a DNA sequence
    @staticmethod
    def get_seq(k, filepath, filetype):
        for seq_record in SeqIO.parse(filepath, filetype):
            # note - only returns sequence from last seq_record if multiple
            rand_start = np.random.randint(0, len(seq_record.seq) - k + 1)
            return seq_record.seq[rand_start:rand_start + k], seq_record.id


def main():

    # set parameters
    n = 1
    num_test_seqs = 2
    len_test_seqs = 150

    # get and vectorize database sequences
    # db_lib = {}
    # for filename in os.listdir('Genomes'):
    #     if filename.endswith('.gbff') and not filename.endswith('test.gbff'):
    #         filename = os.path.join('Genomes', filename)
    #         total_len = 0
    #         org_vec = np.zeros(5**n)
    #         for accession in SeqIO.parse(filename, 'genbank'):
    #             seq = accession.seq
    #             total_len += len(accession.seq)
    #             acc_vec = VectorizeMethods.make_ngram(seq, n)
    #             org_vec += acc_vec
    #         db_lib[filename.split('/')[1].split('.')[0]] = org_vec/(total_len - n + 1)
    # print(db_lib)

    # db_lib = {'Corynebacterium_glutamicum_ASM1132v1': [0.23096325,  0.2703033 ,  0.23093424,  0.26779922,  0.],
    #           'Mycobacterium smegmatis_ASM1500v1': [0.16315325,  0.33732191,  0.16281869,  0.33670616,  0.],
    #           'Streptococcus_pyogenes_ASM678v2': [0.30896556,  0.19074266,  0.3059166 ,  0.19437518,  0.],
    #           'Thermosynechococcus_elongatus_ASM1134v1': [0.23089399,  0.26937183,  0.22992671,  0.26980747,  0.]}

    org_count = 0
    for filename in os.listdir('Genomes'):
        if filename.endswith('.gbff'):
            org_count += 1

    org_num = 0
    X = []
    y = []
    for filename in os.listdir('Genomes'):
        if filename.endswith('.gbff'):
            filename = os.path.join('Genomes', filename)
            for i in range(0, num_test_seqs):
                test_seq, test_id = GenerateSeq.get_seq(len_test_seqs, filename, 'genbank')
                test_vec = VectorizeMethods.make_ngram(test_seq, n)/(len(test_seq) - n + 1)
                X.append(test_vec)
                y.append(test_id)
            org_num += 1

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

    # classify sequence by nearest neighbor in database
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
    print('with n = {}'.format(n))
    print('with num_test_seqs = {}'.format(num_test_seqs))


if __name__ == '__main__':
    main()