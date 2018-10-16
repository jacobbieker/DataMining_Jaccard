import numpy as np
import os
import sys
import scipy.sparse as sparse


"""
This project has to do the following:

python main.py 1234/home/wojtek/A3/evaluation/user_movie.npy

Need to implement minshashing and LSH

Jaccard Similarity

csc and csr

row for minhasing

columns for similarity check



"""

def minhashing(data):
    return NotImplementedError

def lsh(minhashed):
    """
    Takes minhashed values and computes LSH for the values
    :param minhashed:
    :return:
    """
    return NotImplementedError

def write_file(data):
    return NotImplementedError

def calculate_similarity(data):
    minhashed_data = minhashing(data)
    lsh_output = lsh(minhashed_data)

    return NotImplementedError

def convert_data(data):
    """
    Converts the data storage in the .npy file to a sparse matrix
    :param data:
    :return: Compressed Sparse Row matrix for use in minhashing
    """

    # Get unique values for user and movies
    num_users = np.max(data[:,0])+1
    num_movies = np.max(data[:,1])+1

    matrix_values = np.ones(data.shape[0])

    csr_matrix = sparse.csr_matrix((matrix_values, (data[:,0], data[:,1])), shape=(num_users, num_movies))

    # Now get the ones for each element
    return csr_matrix


if __name__ == "__main__":
    # Run when used
    # Get arguments
    arguments = sys.argv
    if len(arguments) < 3:
        # Not long enough
        print("Not enough arguments, input should be of form:\n python main.py seed path\\to\\user_movie.npy")
        exit()
    else:
        # Set seed for random generator
        np.random.seed(int(arguments[1]))
        data = convert_data(np.load(arguments[2]))
        calculate_similarity(data)