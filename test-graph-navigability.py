#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
from tqdm import tqdm
import random
import os
from algs.navigable_graph import NavigableGraph
from algs.k_graph import KGraph
from algs.hnsw import HNSW, heuristic

random.seed(108)

def l2_distance(a, b):
        return np.linalg.norm(a - b)

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

def read_edge_list(file):
    """
    Читает файл с представлением графа в виде списка смежности
    и возвращает граф в виде списка списков.

    :param file: Путь к файлу или файловый объект
    :return: Список списков, представляющий граф
    """
    
    # Открываем файл для чтения
    # with open(file, 'r') as f:
    return [list(map(int, line.strip().split())) for line in file]

def load_vectors_from_file(filename):
    """
    Load vectors from a file where each line contains a space-separated vector.

    :param filename: Path to the file containing vectors
    :return: List of vectors (each vector is a list of floats)
    """
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            # Strip any leading/trailing whitespace and split by spaces
            vector = np.array(list(map(float, line.strip().split())))

            vectors.append(vector)
    return vectors
            
def load_groundtruth_from_file(filename):
    """
    Load ground truth from a file where each line contains space-separated indices.

    :param filename: Path to the file containing ground truth
    :return: List of lists of integers
    """
    groundtruth = []
    with open(filename, 'r') as f:
        for line in f:
            indices = list(map(int, line.strip().split()))
            groundtruth.append(indices)
    return groundtruth

def main():
    parser = argparse.ArgumentParser(description='Test recall of beam search method with KGraph.')
    parser.add_argument('-v', required=True, help='Path to vector file with the vectors')
    parser.add_argument('-t', required=True, help='Path to file with queries')
    parser.add_argument('-gt', required=True, help='Path to ground truth file.  Line i stores indexes of k-nearest neighbors for query i')

    # parser.add_argument('-g', required=True, help='Path to file with the list of neighbors')
    # parser.add_argument('--nq', type=int, default=50, help='Number of query points')
    # TODO тут ещё надо подмать. Может ограничиться тем, что поиск начинается в вершине 0.
    # или пусть они сами выбирают откуда начать поиск?
    # parser.add_argument('--m', type=int, default=3, help='Number of random entry points.') 
    
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')   

    args = parser.parse_args()
    if not os.path.exists(args.v):
        raise FileNotFoundError(f"File with vectors '{args.v}' not found")
    
    if not os.path.exists(args.t):
        raise FileNotFoundError(f"File with queries '{args.t}' not found")
    
    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"File with ground truth '{args.gt}' not found")

    # if not os.path.exists(args.g):
    #     raise FileNotFoundError(f"File with a graph '{args.g}' not found")
    
    # Load vectors and queries
    data = load_vectors_from_file(args.v)
    queries = load_vectors_from_file(args.t)
    gt = load_groundtruth_from_file(args.gt)
    dim = len(data[0])
    k = len(gt[0])
    
    # edge_list = None
    # with open(args.g, 'r') as file:
        # edge_list = read_edge_list(file)

    # graph = NavigableGraph(edge_list, )
    
    # graph = KGraph(k=64, dim=dim, dist_func=KGraph.l2_distance, data=data)

    # Add data to HNSW
    hnsw = HNSW( distance_func=l2_distance, m=32, m0=64, ef=10, ef_construction=5,  neighborhood_construction = heuristic)
    print("Building HNSW graph...")
    for x in tqdm(data):
        hnsw.add(x)

    graph = NavigableGraph(edges=hnsw.get_plane_graph(), points=data, distance_func=l2_distance)   
    
    # Calculate recall
    recall, avg_cal = calculate_recall(graph, queries, gt, k, ef=args.ef, m=10)
    print(f"Average recall: {recall}, avg calc: {avg_cal}")

if __name__ == "__main__":
    main()