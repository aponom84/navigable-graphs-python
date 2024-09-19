#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random
import itertools
random.seed(108)

class KmGraph(object):
    def __init__(self, k, M, dim, dist_func, data):
        self.distance_func = dist_func
        self.k = k
        self.dim = dim
        self.count_brute_force_search = 0
        self.count_greedy_search = 0
        self.data = data
        self.M = M # number of random edges
        # build k-graph by brute force knn-search
        print('Building k-graph')
        self.edges = []
        for x in tqdm(self.data):
            self.edges.append(self.brute_force_knn_search(self.k+1, x)[1:])


        for s, t in random.sample( list(itertools.combinations(range(len(data)), 2)), M ):
            self.edges[s].append( (t, dist_func(data[s], data[t]) ) )

        # self.reset_counters()

    def beam_search(self, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        '''
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = dict() # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate( len(visited), self.data[current_vertex] )

            # check stop conditions #####
            observed_sorted = sorted( observed.items(), key=lambda a: a[1] )
            # print(observed_sorted)
            ef_largets = observed_sorted[ min(len(observed)-1, ef-1 ) ]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor, _ in self.edges[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])

        # Sort the results by distance and return top-k
        observed_sorted =sorted( observed.items(), key=lambda a: a[1] )
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]

    def reset_counters(self):
        self.count_brute_force_search = 0
        self.count_greedy_search = 0

    def l2_distance(a, b):
        return np.linalg.norm(a - b)
    def _vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def brute_force_knn_search(self, k, x):
        '''
        Return the list of (idx, dist) for k-closest elements to {x} in {data}
        '''
        self.count_brute_force_search = self.count_brute_force_search + 1
        return sorted(enumerate(self._vectorized_distance(x, self.data)), key=lambda a: a[1])[:k]

    def plot_graph(self, ax, color, linewidth=0.5):
        ax.scatter(self.data[:, 0], self.data[:, 1], c=color)
        for i in range(len(self.data)):
            for edge_end in self.edges[i]:
                ax.plot( [self.data[i][0], self.data[edge_end][0]], [self.data[i][1], self.data[edge_end][1]], c=color, linewidth=linewidth )

class KGraph(object):
    def __init__(self, k, dim, dist_func, data):
        self.distance_func = dist_func
        self.k = k
        self.dim = dim
        self.count_brute_force_search = 0
        self.count_greedy_search = 0
        self.data = data
        # build k-graph by brute force knn-search
        print('Building k-graph')
        self.edges = []
        for x in tqdm(self.data):
            self.edges.append(self.brute_force_knn_search(self.k+1, x)[1:])


        self.reset_counters()

    def beam_search(self, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        '''
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = dict() # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate( len(visited), self.data[current_vertex] )

            # check stop conditions #####
            observed_sorted = sorted( observed.items(), key=lambda a: a[1] )
            # print(observed_sorted)
            ef_largets = observed_sorted[ min(len(observed)-1, ef-1 ) ]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor, _ in self.edges[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])

        # Sort the results by distance and return top-k
        observed_sorted =sorted( observed.items(), key=lambda a: a[1] )
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]

    def reset_counters(self):
        self.count_brute_force_search = 0
        self.count_greedy_search = 0

    def l2_distance(a, b):
        return np.linalg.norm(a - b)
    def _vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def brute_force_knn_search(self, k, x):
        '''
        Return the list of (idx, dist) for k-closest elements to {x} in {data}
        '''
        self.count_brute_force_search = self.count_brute_force_search + 1
        return sorted(enumerate(self._vectorized_distance(x, self.data)), key=lambda a: a[1])[:k]

    def plot_graph(self, ax, color, linewidth=0.5):
        ax.scatter(self.data[:, 0], self.data[:, 1], c=color)
        for i in range(len(self.data)):
            for edge_end in self.edges[i]:
                ax.plot( [self.data[i][0], self.data[edge_end][0]], [self.data[i][1], self.data[edge_end][1]], c=color, linewidth=linewidth )


def calculate_recall(kg, test, groundtruth, k, ef, m):
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [ [idx for idx, dist in kg.brute_force_knn_search(k, query)] for query in tqdm(test)]

    print("Calculating recall...")
    recalls = []
    total_calc = 0
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        entry_points = random.sample(range(len(kg.data)), m)
        observed = [neighbor for neighbor, dist in kg.beam_search(query, k, entry_points, ef, return_observed = True)]
        total_calc = total_calc + len(observed)
        results = observed[:k]
        intersection = len(set(true_neighbors).intersection(set(results)))
        # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls), total_calc/len(test)


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

    # Create KGraph
    kg = KmGraph(k=args.K, dim=args.dim, dist_func=KGraph.l2_distance, data=train_data, M=args.M)

    # Calculate recall
    recall, avg_cal = calculate_recall(kg, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m)
    print(f"Average recall: {recall}, avg calc: {avg_cal}")


if __name__ == "__main__":
    main()
