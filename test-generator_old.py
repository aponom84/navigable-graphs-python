# not working at the present moment

import numpy as np
import argparse
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if not vec_size:
                break
            vec = np.fromfile(f, dtype=np.float32, count=vec_size[0])
            yield vec


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if not vec_size:
                break
            vec = np.fromfile(f, dtype=np.int32, count=vec_size[0])
            yield vec


def load_sift_dataset():
    train_file = 'datasets/siftsmall/siftsmall_base.fvecs'
    test_file = 'datasets/siftsmall/siftsmall_query.fvecs'
    groundtruth_file = 'datasets/siftsmall/siftsmall_groundtruth.ivecs'

    train_data = np.array(list(read_fvecs(train_file)))
    test_data = np.array(list(read_fvecs(test_file)))
    groundtruth_data = np.array(list(read_ivecs(groundtruth_file)))

    return train_data, test_data, groundtruth_data

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
    data = np.random.random((n, dim)).astype(np.float32)
    queries = np.random.random((nq, dim)).astype(np.float32)

    groundtruth = [ [idx for idx, dist in brute_force_knn_search(k, query, data)] for query in tqdm(queries)]
    return data, queries, groundtruth


def main():
    parser = argparse.ArgumentParser(description='Test recall of beam search method with KGraph.')
    parser.add_argument('--dataset', choices=['synthetic', 'sift'], default='synthetic', help="Choose the dataset to use: 'synthetic' or 'sift'.")
    parser.add_argument('--K', type=int, default=5, help='The size of the neighbourhood')
    parser.add_argument('--M', type=int, default=50, help='Number of random edges')
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data (ignored for SIFT).')
    parser.add_argument('--n', type=int, default=200, help='Number of training points for synthetic data (ignored for SIFT).')
    parser.add_argument('--nq', type=int, default=50, help='Number of query points for synthetic data (ignored for SIFT).')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search in the test stage')
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    parser.add_argument('--m', type=int, default=3, help='Number of random entry points.')

    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'sift':
        print("Loading SIFT dataset...")
        train_data, test_data, groundtruth_data = load_sift_dataset()
    else:
        print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
        train_data, test_data = generate_synthetic_data(args.dim, args.n, args.nq)
        groundtruth_data = None
    

    # Calculate recall
    recall, avg_cal = calculate_recall(kg, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m)
    print(f"Average recall: {recall}, avg calc: {avg_cal}")


if __name__ == "__main__":
    main()