import numpy as np
import os
import sys
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

    signature_matrix = np.zeros((signature, num_users), dtype='int16')
    print(signature_matrix.shape)
    #print("CSR Shape: " + str(csr_matrix.shape))

    # now get the 120 permutations for the signatures
    for permutation in range(signature):
        #print("Permutation: " + str(permutation))
        row_order = np.random.permutation(np.arange(num_movies))
        #print(row_order.shape)
        #print(row_order)

        permuted_csr_matrix = csr_matrix[row_order, :]

        # Now get the first (or last) 1 in each column
        # Number of columns should be the number of users
        for i in range(num_users):
            # This chooses one column at a time and gets the min index for it?
            first = permuted_csr_matrix.indices[permuted_csr_matrix.indptr[i]:permuted_csr_matrix.indptr[i + 1]].min()
            signature_matrix[permutation, i] = first

    # convert to csr matrix because of large number of zeros present still...
    test_sig = sparse.csr_matrix(signature_matrix)

    # Free up memory

    print("Size of full Sig Matrix: " + str(sys.getsizeof(signature_matrix)))
    print("Size of CSR Matrix: " + str(sys.getsizeof(test_sig)))

    print("Num Zeros: " + str(test_sig.nnz))
    print(signature_matrix)

    #del signature_matrix

    return signature_matrix, signature


def lsh(sig_mat, signature, num_bands, sparse_matrix):
    """
    The hash vales are represented as a column, so a new matrix M is the signature matrix where users are still the columns
    but now the rows are the signatures, so from a (100000, 17770) matrix to a (120, 17770) matrix

    Then LSH breaks the signature matrix into b bands of r rows each, and hashes those to the buckets somehow

    Takes minhashed values and computes LSH for the values
    :param minhashed:
    :return:
    """

    bucket = []

    num_rows = int(np.floor(signature / num_bands))
    sparse_matrix = sparse_matrix.toarray()

    # Go through each band
    current_row = 0
    total_number_of_found_pairs = 0
    for bands in range(num_bands):

        # These are the one in the band, so good for csr matrix
        band = sig_mat[current_row:num_rows + current_row, :]
        current_row += num_rows

        # Create the buckets
        print("Band Maxes: " + str(band.max(1) + 1))

        indexes = np.ravel_multi_index(band, band.max(1) + 1)
        s_indexes = indexes.argsort()
        sorted_indexes = indexes[s_indexes]

        bucket_array = np.array(np.split(s_indexes, np.nonzero(sorted_indexes[1:] > sorted_indexes[:-1])[0] + 1))


        # Sort buckets by their length, skip those with length < 2, and work up to larger numbers


        # Only get buckets with more than one user
        for position in range(len(bucket_array)):
            if len(bucket_array[position]) > 1:
                bucket.append(bucket_array[position])

        # Remove buckets with same tuples

        x = map(tuple, bucket)
        bucket = set(x)
        bucket = list(bucket)

        # Largest bucket size
        def lengths(x):
            if isinstance(x,list):
                yield len(x)
                for y in x:
                    yield from lengths(y)

        big_bucket = 0
        for position in range(len(bucket)):
            if len(bucket[position]) > big_bucket:
                big_bucket = len(bucket[position])
        print("\nLargeset Bucket: " + str(big_bucket))

        # sorted by longest length buckets:
        bucket = sorted(bucket, key=len)

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
            # Count how many buckets both pairs have in common vs total number of buckets to get the answer
            for j in large_user_pair:
                # Check if already in unique_set
                if j[0] < j[1]:
                    continue
                else:
                    # In the wrong order
                    j = (j[1], j[0])
                if j not in unique_set:
                    # This is a much faster check of the similarity, not always accurate though
                    sim = signature_similarity(
                        j[0], j[1], sig_mat)
                    if sim > 0.5:
                        # Much more time consuming, but makes sure it is actually higher than 0.5
                        j_sim = jaccards_similarity(j[0], j[1], sparse_matrix)
                        if j_sim > 0.5:
                            unique_set.add(j)

                # Now write out for every 10 that are found
                if len(unique_set) > total_number_of_found_pairs + 10:
                    print("Writing File")
                    write_file(unique_set)
                    total_number_of_found_pairs += 10

    # Also write it when its all done
    write_file(unique_set)

    print(len(unique_set))
    return unique_set

def write_file(unique_set):
    # write to txt file
    unique_set = sorted(unique_set)
    with open('results1.txt', 'w') as f:
        f.write('\n'.join('%s,%s' % user_pair for user_pair in unique_set))
    f.close()
    print('User-Pair found: ', len(unique_set))
    return NotImplementedError

def calculate_similarity(data):
    minhashed_data = minhashing(data)
    lsh_output = lsh(minhashed_data)

    return NotImplementedError


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
        unique_set = lsh(sig_mat, signature, num_bands=19, sparse_matrix=data)
        #calculate_similarity(data)
        print("\nTime Taken: %.2s minutes" % ((time.clock() - start_time) / 60))

