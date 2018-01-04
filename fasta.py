""" generate fasta sequence from data file """
import os
import sys
import numpy


def generate_fasta():
    """
    Generate fasta file for row
    """
    acids_collection = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', '']

    numpy_file = os.path.abspath("../../../Desktop/data/cullpdb+profile_6133.npy.gz")

    cullpdb = numpy.load(numpy_file)

    # row number and column number
    r_num = cullpdb.shape[0]
    c_num = cullpdb.shape[1]

    for i in range(r_num):
        row = cullpdb[i]

        l = range(c_num)
        # get the start points of 700 amino acids
        start_points = l[0::57]
        fasta_sequence = ''

        for p in start_points:
            # only need first 22 char for every amino acid
            segment = row[p:p+22]
            # convert from one hot encoding to letter according to 'acids_collection'
            segment_list = segment.tolist()
            try:
                location_of_one = segment_list.index(1)
                acid = acids_collection[location_of_one]
            except ValueError:
                location_of_one = -1
                acid = ''
            fasta_sequence += acid

        # write seq title and seq to file
        str_i = str(i)
        filename = 'fasta/' + str_i + '.fasta'
        file = open(filename, "w")
        file.write(">seq" + str_i + "\n")
        file.write(fasta_sequence)
        file.close()


def compute_pssm():
    """
    Compute PSSM
    """
    return


def main():
    """ 
    main function 
    """
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'fasta':      ## generate fasta file
            generate_fasta()
        elif arg == 'pssm':     ## compute pssm
            compute_pssm()


if __name__ == '__main__':
    main()
