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
    return NotImplementedError

def write_file(data):
    return NotImplementedError

def calculate_similarity(data):
    minhashed_data = minhashing(data)
    lsh_output = lsh(minhashed_data)

    return NotImplementedError


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
        np.random.seed(arguments[1])
        data = np.load(arguments[2])
