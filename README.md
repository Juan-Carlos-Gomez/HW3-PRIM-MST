# HW3 – Prim’s Algorithm (Minimum Spanning Tree)

J. Carlos Gomez

This project implements **Prim’s algorithm** to construct a **Minimum Spanning Tree (MST)** from an undirected weighted graph represented as an adjacency matrix.  

Additional unit tests were added to verify correctness and robustness.


## Implementation Details

### Prim’s Algorithm
- Implemented in `Graph.construct_mst()` using a priority queue (`heapq`).
- Produces an adjacency matrix representation of the MST.
- Assumes the input graph is undirected and connected.

The MST produced:
- Has exactly `n − 1` edges
- Is connected
- Contains no cycles
- Uses only edges present in the original graph
- Minimizes total edge weight



## Unit Tests

### Provided Tests
- Small graph test (`small.csv`)
- Single-cell distance graph test (Slingshot dataset)

### Student-Added Test
- Custom graph with multiple valid MSTs (tie-handling test)

### Enhanced MST Validation
The `check_mst()` function checks:
- Correct total weight
- Symmetry of adjacency matrices
- No self-loops
- Exactly `n − 1` edges
- All edges exist in the original graph
- Connectivity
- No cycles (Union-Find)

