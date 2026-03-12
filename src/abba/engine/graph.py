"""Graph analysis engine for team network metrics.

Uses scipy.sparse for shortest paths and eigenvalue computation.
Models player relationships as an undirected weighted graph and computes
centrality measures to identify key players and team cohesion.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh


class GraphEngine:
    """Team network analysis using graph theory with scipy."""

    def analyze_team(self, team_data: dict[str, Any]) -> dict[str, Any]:
        """Full team graph analysis."""
        players = team_data.get("players", [])
        relationships = team_data.get("relationships", [])

        if not players or len(players) < 2:
            return {"error": "need at least 2 players"}

        n = len(players)
        adj = np.zeros((n, n), dtype=np.float64)

        for rel in relationships:
            i = rel.get("player1_idx", 0)
            j = rel.get("player2_idx", 0)
            w = rel.get("weight", 1.0)
            if 0 <= i < n and 0 <= j < n and i != j:
                adj[i][j] = w
                adj[j][i] = w

        degree = self._degree_centrality(adj)
        closeness = self._closeness_centrality(adj)
        betweenness = self._betweenness_centrality(adj)
        eigenvector = self._eigenvector_centrality(adj)
        density = self._density(adj)
        clustering = self._clustering_coefficient(adj)
        cohesion = (density + clustering) / 2.0

        # Combined centrality: weighted average of all four measures
        combined = 0.3 * degree + 0.25 * betweenness + 0.25 * eigenvector + 0.2 * closeness
        cmax = combined.max()
        if cmax > 0:
            combined = combined / cmax

        threshold = np.percentile(combined, 70)
        key_indices = [int(i) for i in np.where(combined >= threshold)[0]]

        player_metrics = []
        for i, p in enumerate(players):
            name = p if isinstance(p, str) else p.get("name", f"player_{i}")
            player_metrics.append({
                "name": name,
                "degree_centrality": round(float(degree[i]), 4),
                "closeness_centrality": round(float(closeness[i]), 4),
                "betweenness_centrality": round(float(betweenness[i]), 4),
                "eigenvector_centrality": round(float(eigenvector[i]), 4),
                "combined_score": round(float(combined[i]), 4),
                "is_key_player": i in key_indices,
            })

        return {
            "player_count": n,
            "relationship_count": len(relationships),
            "network_density": round(density, 4),
            "clustering_coefficient": round(clustering, 4),
            "team_cohesion": round(cohesion, 4),
            "key_player_count": len(key_indices),
            "players": player_metrics,
        }

    def _degree_centrality(self, adj: np.ndarray) -> np.ndarray:
        """Normalized sum of edge weights per node."""
        n = len(adj)
        if n <= 1:
            return np.zeros(n)
        degrees = adj.sum(axis=1)
        max_deg = degrees.max()
        return degrees / max_deg if max_deg > 0 else degrees

    def _closeness_centrality(self, adj: np.ndarray) -> np.ndarray:
        """Closeness centrality using scipy shortest_path (Dijkstra).

        C(v) = (n-1) / sum(d(v, u) for all reachable u)
        """
        n = len(adj)
        closeness = np.zeros(n)

        # Convert to sparse graph; scipy expects weights as distances,
        # but our adjacency has weights where higher = stronger connection.
        # For shortest path, invert: distance = 1/weight.
        binary = (adj > 0).astype(np.float64)
        # Use unweighted distances for closeness (standard)
        sparse = csr_matrix(binary)
        dist_matrix = shortest_path(sparse, directed=False, unweighted=True)

        for i in range(n):
            reachable = dist_matrix[i][dist_matrix[i] < np.inf]
            total_dist = reachable.sum()
            if total_dist > 0 and len(reachable) > 1:
                closeness[i] = (len(reachable) - 1) / total_dist

        cmax = closeness.max()
        return closeness / cmax if cmax > 0 else closeness

    def _betweenness_centrality(self, adj: np.ndarray) -> np.ndarray:
        """Betweenness centrality using Brandes' algorithm.

        Counts fraction of shortest paths through each node.
        """
        n = len(adj)
        betweenness = np.zeros(n)
        binary = (adj > 0).astype(np.float64)

        for s in range(n):
            # BFS from s
            stack: list[int] = []
            predecessors: list[list[int]] = [[] for _ in range(n)]
            sigma = np.zeros(n)
            sigma[s] = 1
            dist = np.full(n, -1)
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in range(n):
                    if binary[v][w] == 0:
                        continue
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            delta = np.zeros(n)
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # Normalize for undirected graph
        if n > 2:
            betweenness = betweenness / ((n - 1) * (n - 2))
        bmax = betweenness.max()
        return betweenness / bmax if bmax > 0 else betweenness

    def _eigenvector_centrality(self, adj: np.ndarray) -> np.ndarray:
        """Eigenvector centrality using scipy eigenvalue decomposition.

        The eigenvector corresponding to the largest eigenvalue of the
        adjacency matrix gives each node's centrality.
        """
        n = len(adj)
        if n < 2:
            return np.zeros(n)

        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = eigh(adj)

        # Largest eigenvalue is the last one
        largest_eigenvector = eigenvectors[:, -1]

        # Take absolute values (eigenvector can be negative)
        centrality = np.abs(largest_eigenvector)

        cmax = centrality.max()
        return centrality / cmax if cmax > 0 else centrality

    def _density(self, adj: np.ndarray) -> float:
        """Network density: actual edges / possible edges."""
        n = len(adj)
        max_edges = n * (n - 1) / 2
        if max_edges == 0:
            return 0.0
        actual = np.sum(adj > 0) / 2
        return float(actual / max_edges)

    def _clustering_coefficient(self, adj: np.ndarray) -> float:
        """Average local clustering coefficient.

        For each node, count triangles through it / possible triangles.
        """
        n = len(adj)
        if n < 3:
            return 0.0

        binary = (adj > 0).astype(np.float64)

        # Matrix method: A^3 diagonal gives 2x triangle count per node
        # Number of triangles through node i = (A^3)[i,i] / 2
        a2 = binary @ binary
        a3_diag = np.diag(a2 @ binary)

        coefficients = np.zeros(n)
        for i in range(n):
            k = int(binary[i].sum())  # degree
            if k < 2:
                continue
            possible = k * (k - 1)  # directed pairs among neighbors
            coefficients[i] = a3_diag[i] / possible if possible > 0 else 0.0

        return float(np.mean(coefficients))
