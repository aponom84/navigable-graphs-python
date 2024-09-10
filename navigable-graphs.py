#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import random
from tqdm import tqdm
from heapq import heappush
import time


class KGraph(object): 
    def __init__(self, k, dim, dist_func, data):
        self.distance_func = dist_func
        self.k = k
        self.dim = dim
        self.count_brute_force_search = 0
        self.count_greedy_search = 0
        self.data = data
        print('Building k-graph')
        self.edges = []
        for x in tqdm(self.data):
            self.edges.append(self.brute_force_knn_search(self.k+1, x)[1:])
        self.reset_counters()

    def beam_search(self, q, k, eps, ef):
        candidates = []
        visited = set()
        observed = dict()

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            dist, current_vertex = heappop(candidates)

            observed_sorted = sorted(observed.items(), key=lambda a: a[1])
            ef_largest = observed_sorted[min(len(observed)-1, ef-1)]
            if ef_largest[1] < dist:
                break

            visited.add(current_vertex)

            for neighbor, _ in self.edges[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])                    
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist
                    
        observed_sorted = sorted(observed.items(), key=lambda a: a[1])
        return observed_sorted[:k]

    def reset_counters(self):
        self.count_brute_force_search = 0
        self.count_greedy_search = 0
    
    @staticmethod
    def l2_distance(a, b):
        return np.linalg.norm(a - b)

    def _vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def brute_force_knn_search(self, k, x):
        self.count_brute_force_search += 1
        return sorted(enumerate(self._vectorized_distance(x, self.data)), key=lambda a: a[1])[:k]


def calculate_recall(kg, test, groundtruth, k, ef, M):
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [kg.brute_force_knn_search(k, query) for query in tqdm(test)]

    print("Calculating recall...")
    recalls = []
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        entry_points = random.sample(range(len(kg.data)), M)
        searched_neighbors = [neighbor for neighbor, dist in kg.beam_search(query, k, entry_points, ef)]
        intersection = len(set(true_neighbors).intersection(set(searched_neighbors)))
        recall = intersection / k
        recalls.append(recall)
    
    return np.mean(recalls)


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


def generate_synthetic_data(dim, n, nq):
    train_data = np.random.random((n, dim)).astype(np.float32)
    test_data = np.random.random((nq, dim)).astype(np.float32)
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Test recall of beam search method with KGraph.')
    parser.add_argument('--dataset', choices=['synthetic', 'sift'], default='synthetic', help="Choose the dataset to use: 'synthetic' or 'sift'.")
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data (ignored for SIFT).')
    parser.add_argument('--n', type=int, default=200, help='Number of training points for synthetic data (ignored for SIFT).')
    parser.add_argument('--nq', type=int, default=50, help='Number of query points for synthetic data (ignored for SIFT).')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search for.')
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    parser.add_argument('--M', type=int, default=3, help='Number of random entry points.')
    
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'sift':
        print("Loading SIFT dataset...")
        train_data, test_data, groundtruth_data = load_sift_dataset()
    else:
        print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
        train_data, test_data = generate_synthetic_data(args.dim, args.n, args.nq)
        groundtruth_data = None

    # Create KGraph
    kg = KGraph(k=args.k, dim=args.dim, dist_func=KGraph.l2_distance, data=train_data)

    # Calculate recall
    recall = calculate_recall(kg, test_data, groundtruth_data, k=args.k, ef=args.ef, M=args.M)
    print(f"Average recall: {recall}")


if __name__ == "__main__":
    main()
