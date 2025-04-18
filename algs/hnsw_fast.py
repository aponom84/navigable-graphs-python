import numpy as np
import random
from math import log2
from heapq import heappop, heappush, nsmallest

# Use squared Euclidean distance for efficiency
def l2_distance_squared(a, b):
    diff = a - b
    return diff.dot(diff)

def heuristic(candidates, curr, k, distance_func, data):
    candidates = sorted(candidates, key=lambda a: a[1])
    result = [candidates[0]]
    added_indices = [candidates[0][0]]
    added_data = [data[candidates[0][0]]]
    
    for c, curr_dist in candidates[1:]:
        c_data = data[c]
        added_data_array = np.vstack(added_data)
        # Using squared distances here for consistency if allowed
        dists = np.sum((added_data_array - c_data)**2, axis=1)
        if curr_dist < dists.min():
            result.append((c, curr_dist))
            added_indices.append(c)
            added_data.append(c_data)
    for c, curr_dist in candidates:
        if len(result) < k and (c not in added_indices):
            result.append((c, curr_dist))
    return result

class HNSW:
    def __init__(self, distance_func=l2_distance_squared, m=5, ef=10, ef_construction=30, m0=None, neighborhood_construction=heuristic, vectorized=False):
        self.data = []
        self.distance_func = distance_func
        self.neighborhood_construction = neighborhood_construction
        self._m = m
        self._ef = ef
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef
        point = self._enter_point
        idx = len(self.data)
        self.data.append(elem)
        
        if point is not None:
            d = self.distance_func(elem, self.data[point])
            for layer in reversed(self._graphs[-1:]):  # search top layers (for demonstration)
                point, d = self.beam_search(graph=layer, q=elem, k=1, eps=[point], ef=1)[0]
            layer0 = self._graphs[0]
            for layer in reversed(self._graphs[:min(len(self._graphs), int(-log2(random.random()) * self._level_mult) + 1)]):
                level_m = self._m if layer is not layer0 else self._m0
                candidates = self.beam_search(graph=layer, q=elem, k=level_m*2, eps=[point], ef=self._ef_construction)
                point = candidates[0][0]
                neighbors = self.neighborhood_construction(candidates=candidates, curr=idx, k=level_m, distance_func=self.distance_func, data=self.data)
                layer[idx] = neighbors
                for j, d in neighbors:
                    candidates_j = layer[j] + [(idx, d)]
                    neighbors_j = self.neighborhood_construction(candidates=candidates_j, curr=j, k=level_m, distance_func=self.distance_func, data=self.data)
                    layer[j] = neighbors_j
        for i in range(len(self._graphs), int(-log2(random.random()) * self._level_mult) + 1):
            self._graphs.append({idx: []})
            self._enter_point = idx

    def beam_search(self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        candidates = []
        visited = set()
        observed = {}

        for ep in eps:
            d = self.distance_func(q, self.data[ep])
            heappush(candidates, (d, ep))
            observed[ep] = d

        best_threshold = float('inf')
        while candidates:
            curr_dist, current_vertex = heappop(candidates)
            if curr_dist > best_threshold:
                break
            visited.add(current_vertex)
            for neighbor, _ in graph.get(current_vertex, []):
                if neighbor in observed:
                    continue
                d = self.distance_func(q, self.data[neighbor])
                observed[neighbor] = d
                heappush(candidates, (d, neighbor))
                if len(candidates) > ef:
                    beam = nsmallest(ef, candidates)
                    best_threshold = beam[-1][0]
                else:
                    best_threshold = float('inf')
        observed_sorted = sorted(observed.items(), key=lambda a: a[1])
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]

    # Other functions (search, save_graph_plane, etc.) remain largely unchanged.

# Example usage with random data
if __name__ == '__main__':
    hnsw = HNSW(distance_func=l2_distance_squared, m=5, ef=10, ef_construction=30, neighborhood_construction=heuristic)
    n, dim = 1000, 2
    data = np.random.random((n, dim)).astype(np.float32)
    for x in data:
        hnsw.add(x)
