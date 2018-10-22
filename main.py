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

def to_bucket(num_buckets, num_rows, num_bins, signature):
    """
    Takes the signature break and returns the bucket it should be in

    So for this, create a set of buckets of num_bins * num_rows, and place userIDs that have a signature that places them in
    that set, then order the set by the number elements in each one and go from smallest to largest checking the jsim

    Hash could just be split into the smaller chunks the sig and then mod it with number of buckets

    Bands are the number of  chunks of the signature, rows are the number of elements in each band?


    :param num_rows: Number of rows in signature, ie the length of the signature
    :param num_buckets:
    :param signature:
    :return:
    """

    num_bins_in_signature = len(signature) / num_bins
    buckets = []
    bins = signature.array_split(num_bins_in_signature)
    for i in range(num_bins_in_signature-1):
        # Go through and get the "hashed" value from the columns, return the buckets for the signature
        sub_sig = signature[i*num_bins:num_bins*(i+1)]


    return NotImplementedError


def minhashing(csr_matrix, num_users, num_movies):
    """
    Minhashing has to find the first (or last) 1 in the column where a given column is a user

    Minhashing takes a random number of permutations of the rows (movies) of M and determines for each of those
    where the first 1 shows up in a set of hash values

    The hash vales are represented as a column, so a new matrix M is the signature matrix where users are still the columns
    but now the rows are the signatures, so from a (100000, 17770) matrix to a (120, 17770) matrix

    Then LSH breaks the signature matrix into b bands of r rows each, and hashes those to the buckets somehow


    :param csr_matrix:
    :param num_users:
    :param num_movies:
    :return:
    """

    signature = 120

    signature_matrix = np.zeros((signature, num_users))
    print(signature_matrix.shape)
    print("CSR Shape: " + str(csr_matrix.shape))

    # now get the 120 permutations for the signatures
    for permutation in range(signature):
        print("Permutation: " + str(permutation))
        row_order = np.random.permutation(np.arange(num_movies))
        print(row_order.shape)
        print(row_order)

        permuted_csr_matrix = csr_matrix[row_order, :]
        print(permuted_csr_matrix.shape)

        # Now get the first (or last) 1 in each column
        # Number of columns should be the number of users
        print(num_users)
        for i in range(num_movies-1):
            # This chooses one column at a time and gets the min index for it?
            first = permuted_csr_matrix.indices[permuted_csr_matrix.indptr[i]:permuted_csr_matrix.indptr[i + 1]].min()
            signature_matrix[permutation, i] = first

    return signature_matrix, signature


def lsh(sig_mat, signature):
    """
    Takes minhashed values and computes LSH for the values
    :param minhashed:
    :return:
    """

    sig_mat = sparse.csc_matrix(sig_mat)


    return NotImplementedError

"""
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
"""
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
    print(num_users)
    print(num_movies)

    matrix_values = np.ones(data.shape[0])

    csr_matrix = sparse.csr_matrix((matrix_values, (data[:, 1], data[:, 0])), shape=(num_movies, num_users))

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
        minhashing(data, 103703, 17770)
        calculate_similarity(data)

