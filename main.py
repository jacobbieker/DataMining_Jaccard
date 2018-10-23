import numpy as np
import time

start_time = time.clock()
import sys
import scipy.sparse as sparse
import itertools


def minhashing(csc_matrix, num_users, num_movies):
    """
    Does the minhashing on a CSC sparse matrix, returning the dense signature matrix
    :param csc_matrix: CSC matrix with movies as rows, and users and columns
    :param num_users: Number of users
    :param num_movies: Number of movies
    :return: Signature matrix with rows as signatures and columns as users
    """

    signature = 120

    signature_matrix = np.zeros((signature, num_users), dtype='int16')

    # now get the 120 permutations for the signatures
    for permutation in range(signature):
        row_order = np.random.permutation(np.arange(num_movies))

        # While CSC is not ideal for changing rows, this is faster than converting to csr and converting back
        permuted_csc_matrix = csc_matrix[row_order, :]

        # Number of columns should be the number of users
        for i in range(num_users):
            # Gets all the ones in the column, chooses the min index from the list of all indices in a column
            first = permuted_csc_matrix.indices[permuted_csc_matrix.indptr[i]:permuted_csc_matrix.indptr[i + 1]].min()
            signature_matrix[permutation, i] = first

    return signature_matrix, signature


def lsh(sig_mat, signature, num_bands, sparse_matrix):
    """
    LSH takes the signature matrix and "hashes" them into buckets that are then used to find the similarity
    :param sig_mat: Signature Matrix, dense matrix
    :param signature: Length of the signature
    :param num_bands: Number of bands to use
    :param sparse_matrix: The sparse original matrix
    :return: The unique sets of the data
    """

    bucket = []

    num_rows = int(np.floor(signature / num_bands))
    # Make the sparse matrix dense for the jaccard similarity check
    sparse_matrix = sparse_matrix.toarray()

    # Go through each band
    current_row = 0
    unique_set = set()
    total_ones_found = 0
    for bands in range(num_bands):

        # These are the one in the band, so good for csr matrix
        band = sig_mat[current_row:num_rows + current_row, :]
        current_row += num_rows

        # Create the buckets
        indexes = np.ravel_multi_index(band, band.max(1) + 1)
        s_indexes = indexes.argsort()
        sorted_indexes = indexes[s_indexes]

        bucket_array = np.array(np.split(s_indexes, np.nonzero(sorted_indexes[1:] > sorted_indexes[:-1])[0] + 1))

        # Only get buckets with more than one user
        for position in range(len(bucket_array)):
            if len(bucket_array[position]) > 1:
                bucket.append(bucket_array[position])

        # Go through all the buckets, finding the actual similar pairs
        for i in range(len(bucket)):
            # creates a generator to go through all the combinations in a given bucket
            user_pairs = set(pair for pair in itertools.combinations(bucket[i], 2))

            user_pairs = user_pairs.difference(unique_set)
            # Count how many buckets both pairs have in common vs total number of buckets to get the answer
            for pair in user_pairs:
                # Check if already in unique_set
                if pair not in unique_set and (pair[1], pair[0]) not in unique_set:
                    # This is a much faster check of the similarity, not always accurate though, could also eliminate
                    # some truly similar objects, but is much faster, so have lower threshold for this one
                    sim = signature_similarity(pair[0], pair[1], sig_mat)
                    if sim > 0.35:
                        # Much more time consuming, but makes sure it is actually higher than 0.5
                        j_sim = bool_jaccards_similarity(pair[0], pair[1], sparse_matrix)
                        if j_sim > 0.5:
                            if pair[0] < pair[1]:
                                unique_set.add(pair)
                            else:
                                unique_set.add((pair[1], pair[0]))
                            # Now write out as it goes
                            if len(unique_set) > total_ones_found + 10:
                                # Write every 10 as a checkpoint
                                write_file(unique_set)
                                total_ones_found = len(unique_set)

    # Also write it when its all done
    # check Unique set against the real results

    write_file(unique_set)
    return unique_set


def write_file(unique_set):
    # write to txt file
    written_values = 0
    unique_set = sorted(unique_set)
    # Now check if there are duplicates
    with open("results.txt", "w") as f:
        for set in unique_set:
            f.write(str(set[0]) + "," + str(set[1]) + "\n")
            written_values += 1


def signature_similarity(user1, user2, signature_matrix):
    """
    Calculates the similarity in the signature matrix, directly since its much smaller
    :param user1: The first user
    :param user2: The second user
    :param signature_matrix: The dense signature matrix
    :return: The similarity score based on the signature matrix for two users
    """
    sim_score = float(np.count_nonzero(signature_matrix[:, user1] == signature_matrix[:, user2])) \
                / len(signature_matrix[:, user1])
    return sim_score


def bool_jaccards_similarity(user1, user2, dense_matrix):
    """
    Calculates the Jaccard Similarity on a boolean matrix given two users
    :param user1: First user
    :param user2: Second User
    :param dense_matrix: Movie x User Dense matrix to use
    :return: The Jaccard similarity
    """
    # Numerator, the intersection of both users
    intersection = np.logical_and(dense_matrix[:, user1], dense_matrix[:, user2])
    union = np.logical_or(dense_matrix[:, user1], dense_matrix[:, user2])
    jacard_sim = intersection.sum() / float(union.sum())

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

    matrix_values = np.ones(data.shape[0])

    # Used boolean as that saves on memory and allows calculating the Jaccard similarity easier
    csr_matrix = sparse.csc_matrix((matrix_values, (data[:, 1], data[:, 0])), shape=(num_movies, num_users), dtype='b')

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
        sig_mat, signature = minhashing(data, data.shape[1], data.shape[0])
        unique_set = lsh(sig_mat, signature, num_bands=20, sparse_matrix=data)
        print("\nTime Taken: %.2s minutes" % ((time.clock() - start_time) / 60))
