"""Graph analysis for team performance in ABBA."""

from collections import defaultdict
from typing import Any

import numpy as np

from ..core.logging import get_logger

logger = get_logger(__name__)


class GraphAnalyzer:
    """Analyzes team performance using graph theory concepts."""

    def __init__(self):
        """Initialize the graph analyzer."""
        self.centrality_weights = {
            "degree": 0.3,
            "betweenness": 0.3,
            "closeness": 0.2,
            "eigenvector": 0.2,
        }

    async def build_graph(self, team_data: dict[str, Any]) -> dict[str, Any] | None:
        """Build a graph representation of team relationships.

        Args:
            team_data: Team performance and relationship data

        Returns:
            Graph structure or None
        """
        try:
            if not team_data:
                logger.warning("No team data provided")
                return None

            # Extract players and their relationships
            players = team_data.get("players", [])
            relationships = team_data.get("relationships", [])

            if not players:
                logger.warning("No players found in team data")
                return None

            # Build adjacency matrix
            n_players = len(players)
            adjacency_matrix = np.zeros((n_players, n_players))

            # Fill adjacency matrix based on relationships
            for rel in relationships:
                player1_idx = rel.get("player1_idx", 0)
                player2_idx = rel.get("player2_idx", 0)
                weight = rel.get("weight", 1.0)

                if 0 <= player1_idx < n_players and 0 <= player2_idx < n_players:
                    adjacency_matrix[player1_idx][player2_idx] = weight
                    adjacency_matrix[player2_idx][
                        player1_idx
                    ] = weight  # Undirected graph

            # Calculate centrality measures
            centrality = await self._calculate_centrality(adjacency_matrix)

            # Identify key players
            key_players = await self._identify_key_players(players, centrality)

            # Calculate team cohesion
            cohesion = await self._calculate_cohesion(adjacency_matrix)

            graph = {
                "players": players,
                "adjacency_matrix": adjacency_matrix.tolist(),
                "centrality": centrality,
                "key_players": key_players,
                "cohesion": cohesion,
                "relationships": relationships,
            }

            logger.info(
                f"Built graph with {n_players} players and {len(relationships)} relationships"
            )

            return graph

        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return None

    async def analyze_connections(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Analyze connections and network structure.

        Args:
            graph: Graph structure

        Returns:
            Connection analysis results
        """
        try:
            if not graph:
                return {}

            adjacency_matrix = np.array(graph.get("adjacency_matrix", []))
            players = graph.get("players", [])

            if adjacency_matrix.size == 0:
                return {}

            # Calculate connection metrics
            total_connections = np.sum(adjacency_matrix) / 2  # Undirected graph
            avg_connections = total_connections / len(players) if players else 0

            # Calculate clustering coefficient
            clustering = self._calculate_clustering_coefficient(adjacency_matrix)

            # Analyze connection distribution
            connection_distribution = self._analyze_connection_distribution(
                adjacency_matrix
            )

            # Calculate network density
            density = self._calculate_network_density(adjacency_matrix)

            return {
                "total_connections": float(total_connections),
                "avg_connections_per_player": float(avg_connections),
                "clustering_coefficient": float(clustering),
                "network_density": float(density),
                "connection_distribution": connection_distribution,
            }

        except Exception as e:
            logger.error(f"Error analyzing connections: {e}")
            return {}

    async def _calculate_centrality(
        self, adjacency_matrix: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Calculate centrality measures for each player.

        Args:
            adjacency_matrix: Adjacency matrix of the graph

        Returns:
            Dictionary of centrality measures
        """
        try:
            n_players = len(adjacency_matrix)
            centrality = {
                "degree": {},
                "betweenness": {},
                "closeness": {},
                "eigenvector": {},
            }

            # Degree centrality
            degree_centrality = np.sum(adjacency_matrix, axis=1)
            for i in range(n_players):
                centrality["degree"][f"player_{i}"] = float(
                    degree_centrality[i] / (n_players - 1)
                )

            # Closeness centrality (simplified)
            closeness_centrality = self._calculate_closeness_centrality(
                adjacency_matrix
            )
            for i in range(n_players):
                centrality["closeness"][f"player_{i}"] = float(closeness_centrality[i])

            # Betweenness centrality (simplified)
            betweenness_centrality = self._calculate_betweenness_centrality(
                adjacency_matrix
            )
            for i in range(n_players):
                centrality["betweenness"][f"player_{i}"] = float(
                    betweenness_centrality[i]
                )

            # Eigenvector centrality (simplified)
            eigenvector_centrality = self._calculate_eigenvector_centrality(
                adjacency_matrix
            )
            for i in range(n_players):
                centrality["eigenvector"][f"player_{i}"] = float(
                    eigenvector_centrality[i]
                )

            return centrality

        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}

    def _calculate_closeness_centrality(
        self, adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate closeness centrality (simplified).

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Closeness centrality values
        """
        try:
            # Simplified closeness centrality
            # In practice, this would use shortest path algorithms
            degree_centrality = np.sum(adjacency_matrix, axis=1)
            max_degree = np.max(degree_centrality) if degree_centrality.size > 0 else 1

            return (
                degree_centrality / max_degree
                if max_degree > 0
                else np.zeros_like(degree_centrality)
            )

        except Exception as e:
            logger.error(f"Error calculating closeness centrality: {e}")
            return np.zeros(len(adjacency_matrix))

    def _calculate_betweenness_centrality(
        self, adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate betweenness centrality (simplified).

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Betweenness centrality values
        """
        try:
            # Simplified betweenness centrality
            # In practice, this would count shortest paths through each node
            n_players = len(adjacency_matrix)
            betweenness = np.zeros(n_players)

            # Simple approximation based on degree and clustering
            degree_centrality = np.sum(adjacency_matrix, axis=1)
            clustering = self._calculate_clustering_coefficient(adjacency_matrix)

            for i in range(n_players):
                betweenness[i] = degree_centrality[i] * (1 - clustering)

            # Normalize
            max_betweenness = np.max(betweenness) if betweenness.size > 0 else 1
            return betweenness / max_betweenness if max_betweenness > 0 else betweenness

        except Exception as e:
            logger.error(f"Error calculating betweenness centrality: {e}")
            return np.zeros(len(adjacency_matrix))

    def _calculate_eigenvector_centrality(
        self, adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate eigenvector centrality (simplified).

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Eigenvector centrality values
        """
        try:
            # Simplified eigenvector centrality
            # In practice, this would use eigenvalue decomposition
            degree_centrality = np.sum(adjacency_matrix, axis=1)

            # Simple approximation
            eigenvector = degree_centrality.copy()

            # Normalize
            max_eigenvector = np.max(eigenvector) if eigenvector.size > 0 else 1
            return eigenvector / max_eigenvector if max_eigenvector > 0 else eigenvector

        except Exception as e:
            logger.error(f"Error calculating eigenvector centrality: {e}")
            return np.zeros(len(adjacency_matrix))

    async def _identify_key_players(
        self, players: list[Any], centrality: dict[str, dict[str, float]]
    ) -> list[str]:
        """Identify key players based on centrality measures.

        Args:
            players: List of players
            centrality: Centrality measures

        Returns:
            List of key player identifiers
        """
        try:
            if not centrality or not players:
                return []

            # Calculate combined centrality score
            player_scores = defaultdict(float)

            for measure, weights in centrality.items():
                weight = self.centrality_weights.get(measure, 0.25)
                for player_id, score in weights.items():
                    player_scores[player_id] += weight * score

            # Sort by score and return top players
            sorted_players = sorted(
                player_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Return top 30% of players as key players
            n_key_players = max(1, int(len(players) * 0.3))
            key_players = [player_id for player_id, _ in sorted_players[:n_key_players]]

            return key_players

        except Exception as e:
            logger.error(f"Error identifying key players: {e}")
            return []

    async def _calculate_cohesion(self, adjacency_matrix: np.ndarray) -> float:
        """Calculate team cohesion score.

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Cohesion score (0-1)
        """
        try:
            if adjacency_matrix.size == 0:
                return 0.0

            n_players = len(adjacency_matrix)

            # Calculate clustering coefficient
            clustering = self._calculate_clustering_coefficient(adjacency_matrix)

            # Calculate average connection strength
            avg_connection = np.mean(adjacency_matrix)

            # Calculate connection density
            max_connections = n_players * (n_players - 1) / 2
            actual_connections = np.sum(adjacency_matrix > 0) / 2
            density = actual_connections / max_connections if max_connections > 0 else 0

            # Combine metrics
            cohesion = (clustering + avg_connection + density) / 3

            return float(np.clip(cohesion, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating cohesion: {e}")
            return 0.0

    def _calculate_clustering_coefficient(self, adjacency_matrix: np.ndarray) -> float:
        """Calculate clustering coefficient.

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Clustering coefficient
        """
        try:
            if adjacency_matrix.size == 0:
                return 0.0

            # Simplified clustering coefficient
            # In practice, this would count triangles
            n_players = len(adjacency_matrix)

            # Calculate average degree
            degrees = np.sum(adjacency_matrix, axis=1)
            avg_degree = np.mean(degrees)

            # Simple approximation
            clustering = avg_degree / (n_players - 1) if n_players > 1 else 0

            return float(np.clip(clustering, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating clustering coefficient: {e}")
            return 0.0

    def _calculate_network_density(self, adjacency_matrix: np.ndarray) -> float:
        """Calculate network density.

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Network density
        """
        try:
            if adjacency_matrix.size == 0:
                return 0.0

            n_players = len(adjacency_matrix)
            max_edges = n_players * (n_players - 1) / 2

            if max_edges == 0:
                return 0.0

            actual_edges = np.sum(adjacency_matrix > 0) / 2
            density = actual_edges / max_edges

            return float(density)

        except Exception as e:
            logger.error(f"Error calculating network density: {e}")
            return 0.0

    def _analyze_connection_distribution(
        self, adjacency_matrix: np.ndarray
    ) -> dict[str, float]:
        """Analyze the distribution of connections.

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Connection distribution statistics
        """
        try:
            if adjacency_matrix.size == 0:
                return {}

            degrees = np.sum(adjacency_matrix, axis=1)

            return {
                "min_connections": float(np.min(degrees)),
                "max_connections": float(np.max(degrees)),
                "avg_connections": float(np.mean(degrees)),
                "std_connections": float(np.std(degrees)),
                "median_connections": float(np.median(degrees)),
            }

        except Exception as e:
            logger.error(f"Error analyzing connection distribution: {e}")
            return {}
