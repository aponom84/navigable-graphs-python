#!/usr/bin/env python
# coding: utf-8

from algs.navigable_graph import NavigableGraph
import numpy as np
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random

random.seed(108)

class KGraph(NavigableGraph):
    '''
    One can create KGraph
    ```
    kg = KMGraph(k=args.K, dim=args.dim, dist_func=KGraph.l2_distance, data=train_data, M=args.M)
    ```
    '''
    def __init__(self, k, dim, dist_func, data):
        super().__init__(edges=[], points=data)
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
