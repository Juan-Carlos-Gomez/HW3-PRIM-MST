import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray,
              mst: np.ndarray,
              expected_weight: float,
              allowed_error: float = 0.0001):
    """

    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot
    simply check for equality against a known MST of a graph.

    Adds assertions for:
      - shape + symmetry
      - no self loops
      - MST edges must exist in original graph with same weights
      - exactly n-1 edges
      - connected
      - acyclic (tree)
      - correct total weight

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    # ---- basic checks ----
    assert mst is not None, "MST is None (did construct_mst set self.mst?)"
    assert isinstance(adj_mat, np.ndarray) and isinstance(mst, np.ndarray)
    assert adj_mat.shape == mst.shape, "MST must have same shape as adjacency matrix"
    assert adj_mat.ndim == 2 and adj_mat.shape[0] == adj_mat.shape[1], "adj_mat must be square"

    n = adj_mat.shape[0]

    # ---- undirected: symmetric matrices ----
    assert np.allclose(adj_mat, adj_mat.T), "Input adjacency matrix must be symmetric"
    assert np.allclose(mst, mst.T), "MST adjacency matrix must be symmetric"

    # ---- no self-loops ----
    assert np.allclose(np.diag(mst), 0), "MST diagonal must be zero (no self loops)"

    # ---- MST edges must be a subset of original edges (and weights must match) ----
    for i in range(n):
        for j in range(n):
            if mst[i, j] != 0:
                assert adj_mat[i, j] != 0, f"MST contains edge ({i},{j}) that does not exist in graph"
                assert approx_equal(mst[i, j], adj_mat[i, j]), f"MST weight at ({i},{j}) differs from adj_mat"

    # ---- edge count should be n-1 (count undirected edges once: i<j) ----
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if mst[i, j] != 0]
    assert len(edges) == n - 1, f"MST must have n-1 edges; got {len(edges)}"

    # ---- acyclic check (Union-Find) ----
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    for u, v in edges:
        assert union(u, v), f"Cycle detected when adding edge ({u},{v})"

    # ---- connectivity check: DFS on MST ----
    mst_adj = [[] for _ in range(n)]
    for u, v in edges:
        mst_adj[u].append(v)
        mst_adj[v].append(u)

    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        cur = stack.pop()
        for nxt in mst_adj[cur]:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(nxt)

    assert all(seen), "MST is not connected (does not span all nodes)"

    # ---- total weight check (sum lower triangle, includes diagonal=0) ----
    total = 0.0
    for i in range(n):
        for j in range(i + 1):
            total += mst[i, j]

    assert approx_equal(total, expected_weight), "Proposed MST has incorrect expected weight"


def test_mst_small():
    """
    Unit test for the construction of a minimum spanning tree on a small graph.
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.
    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path)
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    New student test: graph with multiple valid MSTs (ties).
    Expected MST total weight = 3.
    """
    # 4-node cycle edges weight=1, diagonals weight=2
    adj = np.array([
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ], dtype=float)

    g = Graph(adj)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, expected_weight=3.0)
