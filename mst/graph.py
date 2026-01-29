import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat`
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.
        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else:
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`.

        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists.

        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the
        `heapify`, `heappop`, and `heappush` functions.

        """
        adj = self.adj_mat
        if not isinstance(adj, np.ndarray) or adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("adjacency matrix must be a square 2D numpy array")

        n = adj.shape[0]
        if n == 0:
            self.mst = np.zeros_like(adj, dtype=float)
            return

        # Prim's algorithm
        mst = np.zeros_like(adj, dtype=float)
        visited = np.zeros(n, dtype=bool)

        # (weight, u, v)
        heap = []

        start = 0
        visited[start] = True
        visited_count = 1
        for v in range(n):
            w = adj[start, v]
            if v != start and w != 0:
                heapq.heappush(heap, (w, start, v))

        edges_added = 0
        while heap and edges_added < n - 1:
            w, u, v = heapq.heappop(heap)

            if visited[v]:
                continue

            # add edge to MST (undirected)
            mst[u, v] = w
            mst[v, u] = w
            edges_added += 1
            visited[v] = True
            visited_count += 1
            for nxt in range(n):
                w2 = adj[v, nxt]
                if nxt != v and (not visited[nxt]) and w2 != 0:
                    heapq.heappush(heap, (w2, v, nxt))

        if visited_count != n:
            raise ValueError("Input graph is disconnected: MST spanning all nodes does not exist.")

        self.mst = mst
