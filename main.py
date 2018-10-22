import numpy as np
import os
import time
start_time = time.clock()
import sys
import scipy.sparse as sparse
import itertools


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

    signature_matrix = np.zeros((signature, num_users), dtype='int32')
    print(signature_matrix.shape)
    #print("CSR Shape: " + str(csr_matrix.shape))

    # now get the 120 permutations for the signatures
    for permutation in range(signature):
        #print("Permutation: " + str(permutation))
        row_order = np.random.permutation(np.arange(num_movies))
        #print(row_order.shape)
        #print(row_order)

        permuted_csr_matrix = csr_matrix[row_order, :]
        #print(permuted_csr_matrix.shape)

        # Now get the first (or last) 1 in each column
        # Number of columns should be the number of users
        #print(num_users)
        for i in range(num_users):
            # This chooses one column at a time and gets the min index for it?
            #print(permuted_csr_matrix.indptr[i])
            #print(permuted_csr_matrix.indices[permuted_csr_matrix.indptr[i]])
            first = permuted_csr_matrix.indices[permuted_csr_matrix.indptr[i]:permuted_csr_matrix.indptr[i + 1]].min()
            signature_matrix[permutation, i] = first

    test_sig = sparse.csr_matrix(signature_matrix)

    print("Num Zeros: " + str(test_sig.nnz))
    print(signature_matrix)

    return signature_matrix, signature

def signature_similarity(user1, user2, signature_matrix):
    """    Calculate and return the similarity between two users (user1 and user2) with the signature matrix    """
    similar = float(np.count_nonzero(signature_matrix[:, user1] == signature_matrix[:, user2])) \
              / len(signature_matrix[:, user1])
    return similar

def jaccards_similarity(user1, user2, sparse_matrix):
    """     Calculate and return the Jaccard similarity between two users (user1 and user2) with the sparse matrix   """
    sum_val = np.sum(sparse_matrix[:, user1] & sparse_matrix[:, user2])
    sim_val = np.sum(sparse_matrix[:, user1] | sparse_matrix[:, user2])
    jacard_sim = float(sum_val) / float(sim_val)
    return jacard_sim

def output(original_sparse, unique_set):
    """
    Create the txt file with the candidate pairs that have a real jaccard similarity larger than 0.5
    :param original_sparse -- the original sparse matrix
    :param unique_set -- the unique set found with banding of which we calculate the jaccard similarity
    """
    print("Outputing results.. ")

    # make from the sparse array a real array, such that we now also put into
    # memory the '0's.
    sparse_array = original_sparse.toarray()

    # order the set on the first element of the tuples, iterating over a list is faster than over a set
    # However the in function is slower with a list, but this is done less
    # often than the iterations
    original_unique_set = unique_set
    unique_set = sorted(unique_set)

    # empty list which we append the found pairs, so that we can sort at the end again; needed because
    #  user2 can > user1
    user_pair_list = []
    # check if the similarity is really > 0.5 with the jaccard similarity and
    # add to the txt file if user1<user2
    for pair in unique_set:
        # if user1 < user2 and jaccard similarity > 0.5 add to txt file
        if pair[0] < pair[1]:
            sim = jaccards_similarity(pair[0], pair[1], sparse_array)
            if sim > 0.5:
                # add +1 to users because we started counting from 0 in python
                user_pair_list.append((pair[0] + 1, pair[1] + 1))

        # if user 2 is larger than user 1 and combine (user2, user1) is already in unique_set continue with loop
        # else calculate similarity
        elif pair[0] > pair[1]:
            # if pair is already in set skip it and go into next iteration for loop; E.g. (3,1) is same as (1,3) thus
            #  skip it
            if (pair[1], pair[0]) in original_unique_set:
                continue
            # if this is note the case add it with user 2 at position user 1, such that the txt file's user1 is always
            # smaller than user 2
            else:
                sim = jaccards_similarity(pair[0], pair[1], sparse_array)
                if sim > 0.5:
                    # add +1 to users because we started counting from 0 in
                    # python
                    user_pair_list.append((pair[1] + 1, pair[0] + 1))

    # sort user pair list on first user
    user_pair_list = sorted(user_pair_list)

    print("Length of user pairs: " + str(len(user_pair_list)))

    # write to txt file
    with open('results.txt', 'w') as f:
        f.write('\n'.join('%s,%s' % user_pair for user_pair in user_pair_list))
    f.close()
    print('User-Pair found: ', len(user_pair_list))

def lsh(sig_mat, signature, num_bands):
    """
    The hash vales are represented as a column, so a new matrix M is the signature matrix where users are still the columns
    but now the rows are the signatures, so from a (100000, 17770) matrix to a (120, 17770) matrix

    Then LSH breaks the signature matrix into b bands of r rows each, and hashes those to the buckets somehow

    Takes minhashed values and computes LSH for the values
    :param minhashed:
    :return:
    """

    #sig_mat = sparse.csc_matrix(sig_mat)

    bucket = []

    num_rows = signature / num_bands

    # Go through each band
    current_row = 0
    for bands in range(num_bands):

        band = sig_mat[current_row:int(num_rows) + current_row, :]
        current_row += int(num_rows)

        # Create the buckets

        indexes = np.ravel_multi_index(band.astype(int), band.max(1).astype(int) + 1)
        s_indexes = indexes.argsort()
        sorted_indexes = indexes[s_indexes]

        bucket_array = np.array(np.split(s_indexes, np.nonzero(sorted_indexes[1:] > sorted_indexes[:-1])[0] + 1))

        # Only get buckets with more than one user
        for position in range(len(bucket_array)):
            if len(bucket_array[position]) > 1:
                bucket.append(bucket_array[position])


        # Remove buckets with same tuples

        x = map(tuple, bucket)
        bucket = set(x)
        bucket = list(bucket)

        # finding the unique candidate pairs with a similarity larger than 0.5 in the signature matrix
        # note that this also counts (3,5) and (5,3) separately. This double counting
        # is removed later on during creation of the txt file
        unique_set = set()
        for i in range(len(bucket)):
            # generate a generator expression for the combinations of the pairs in
            # a bucket
            large_user_pair = set(
                pair for pair in itertools.combinations(bucket[i], 2))

            large_user_pair = large_user_pair.difference(unique_set)
            for j in large_user_pair:
                sim = signature_similarity(
                    j[0], j[1], sig_mat)
                if sim > 0.5:
                    unique_set.add(j)


    print(len(unique_set))
    print(unique_set)
    return unique_set

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

    csr_matrix = sparse.csc_matrix((matrix_values, (data[:, 1], data[:, 0])), shape=(num_movies, num_users), dtype='b')

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
        sig_mat, signature = minhashing(data, 103703, 17770)
        unique_set = lsh(sig_mat, signature, num_bands=20)
        output(data, unique_set)
        #calculate_similarity(data)
        print("\nTime Taken: %.2s minutes" % ((time.clock() - start_time) / 60))

