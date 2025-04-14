#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
from tqdm import tqdm

def l2_distance(a, b):
    return np.linalg.norm(a - b)


def _vectorized_distance(x, ys):
    return [l2_distance(x, y) for y in ys]


def brute_force_knn_search(k, x, data):
    '''
    Return the list of (idx, dist) for k-closest elements to {x} in {data}
    '''
    return sorted(enumerate(_vectorized_distance(x, data)), key=lambda a: a[1])[:k]


def generate_synthetic_data(dim, n, nq, k):
    """
    Generate synthetic training data, query data, and ground truth.
    :param dim: Dimensionality of the vectors
    :param n: Number of training points
    :param nq: Number of query points
    :param k: Number of nearest neighbors to search
    :return: Training data, query data, and ground truth
    """
    # Generate random training data and query data
    data = np.random.random((n, dim)).astype(np.float32)
    queries = np.random.random((nq, dim)).astype(np.float32)

    # Compute ground truth using brute-force KNN search
    groundtruth = [[idx for idx, dist in brute_force_knn_search(k, query, data)] for query in tqdm(queries)]

    return data, queries, groundtruth


def save_vectors_to_file(filename, vectors):
    """
    Save vectors to a file, one vector per line.
    :param filename: Path to the output file
    :param vectors: List of vectors to save
    """
    with open(filename, 'w') as f:
        for vector in vectors:
            f.write(" ".join(map(str, vector)) + "\n")


def save_groundtruth_to_file(filename, groundtruth):
    """
    Save ground truth to a file, one query's neighbors per line.
    :param filename: Path to the output file
    :param groundtruth: List of lists containing neighbor indices
    """
    with open(filename, 'w') as f:
        for neighbors in groundtruth:
            f.write(" ".join(map(str, neighbors)) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Synthetic data generator',
        epilog='''
Example Usage:
  python script.py --dim 128 --n 1000 --nq 100 --k 10 \\
      -v train_vectors.txt \\
      -t query_vectors.txt \\
      --gt groundtruth.txt

This will generate:
  - 1000 training vectors with 128 dimensions and save them to train_vectors.txt
  - 100 query vectors with 128 dimensions and save them to query_vectors.txt
  - Ground truth with the 10 nearest neighbors for each query and save it to groundtruth.txt
''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data')
    parser.add_argument('--n', type=int, default=200, help='Number of training points for synthetic data')
    parser.add_argument('--nq', type=int, default=50, help='Number of query points for synthetic data')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search in the test stage')
    parser.add_argument('-v', required=True, help='Path to vector file with the training vectors. Each vector stored in a single line as plain text')
    parser.add_argument('-t', required=True, help='Path to file with query vectors. Each vector stored in a single line as plain text')
    parser.add_argument('--gt', required=True, help='Path to ground truth file. Line i stores indexes of k-nearest neighbors for query i')

    args = parser.parse_args()

    print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
    train_data, test_data, groundtruth_data = generate_synthetic_data(args.dim, args.n, args.nq, args.k)

    # Save the generated data to the specified files
    print(f"Saving training data to {args.v}")
    save_vectors_to_file(args.v, train_data)

    print(f"Saving query data to {args.t}")
    save_vectors_to_file(args.t, test_data)

    print(f"Saving ground truth to {args.gt}")
    save_groundtruth_to_file(args.gt, groundtruth_data)


if __name__ == "__main__":
    main()