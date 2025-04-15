#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random
import itertools
from algs.navigable_graph import NavigableGraph
random.seed(108)

class KMGraph(NavigableGraph):
    '''
    K-Graph with m random edges
    '''
    
    def __init__(self, k, M, dim, dist_func, data):
        super().__init__(edges=[], points=data, distance_func=dist_func)
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