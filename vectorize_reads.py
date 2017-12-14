from Bio import SeqIO
import numpy as np
import os


class VectorizeMethods:
    def __init__(self):
        pass

    # This function takes in a DNA sequence and returns it as a 5x1 vector
    @staticmethod
    def make_onegram(seq):
        return


class GenerateSeq:
    def __init__(self):
        pass

    # This function takes in a filepath and generates a DNA sequence
    @staticmethod
    def get_150seq(k, filepath, filetype):
        for seq_record in SeqIO.parse(filepath, filetype):
            # note - only returns sequence from last seq_record if multiple
            rand_start = np.random.randint(0, len(seq_record.seq) - k + 1)
            return seq_record.seq[rand_start:rand_start + k], seq_record.id

def main():

    seq, id = GenerateSeq.get_150seq(150, "Bifidobacterium_longum_ASM752v1_test.gbff", "genbank")

if __name__ == '__main__':
    main()