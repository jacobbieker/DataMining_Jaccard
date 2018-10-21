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

Minhashing: shingles are the movies, signature is just finding the 50-150 movies that each person rates, saving it as 
the row/column number for that user, so if user 1 rated movies in movies 1,4,5, his signature would be sig(1,4,5) as each
movie is a row

Those signatures are then hashed to buckets in LSH, and then determines how similar two users are based on the number of
matching shingles -> matching buckets




"""
data = np.load('user_movie.npy')

def minhashing(csr_matrix, num_users, num_movies):

    signature = 120
    sig_mat = np.zeros((signature, num_users))

    # row order
    for u in range(signature):
        row = np.random.permutation(np.arange(num_movies))
        row = tuple(row)
        
        # Swap the sparse rows
        csr_matrix_nw = csr_matrix[row, :]

        # find the first '1' in column
        for i in range(num_users):
            first = csr_matrix_nw.indices[csr_matrix_nw.indptr[i]:csr_matrix_nw.indptr[i + 1]].min()
            sig_mat[u, i] = first

    return sig_mat, signature


def lsh(data):
    """
    Takes minhashed values and computes LSH for the values
    :param minhashed:
    :return:
    """
    return NotImplementedError


def write_file(userU1U2, real_sparse):

    _sparse = real_sparse.toarray()
    lsh_user_pair = pairs
    pairs = sorted(pairs)
    user_pair = []

    for user in pairs:
        if user[0] < user[1]:
            jaccard = calculate_similarity(user[0], user[1], _sparse)
            if jaccard > 0.5:
                user_pair.append((user[0] + 1, user[1] + 1, jaccard))

        if user[0] > user[1]:
            if (user[1], user[0]) in lsh_user_pair:
                continue
        else:
            jaccard = calculate_similarity(user[0], user[1], _sparse)
            if jaccard > 0.5:
                lsh_user_pair.append((user[1] + 1, user[0] + 1, jaccard))

    lsh_user_pair = sorted(lsh_user_pair)

    with open("results.txt", "w") as f:
        f.write("{0}, {1}\n".format(user[0], user[1]))
    f.close()

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
    num_users = np.max(data[:, 0]) + 1
    num_movies = np.max(data[:, 1]) + 1

    matrix_values = np.ones(data.shape[0])

    csr_matrix = sparse.csr_matrix((matrix_values, (data[:, 0], data[:, 1])), shape=(num_users, num_movies))

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
